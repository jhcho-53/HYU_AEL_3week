"""notebooks/real_ipm_lab.ipynb 생성기. 노트북 내용의 원본은 이 파일임.

노트북 고칠 일 있으면 .ipynb 말고 여기를 고치고 다시 생성할 것 (직접 고치면 다음 생성 때 덮어써짐).

    python camsim/scripts/build_real_notebook.py
"""
import json
import os
import textwrap

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "id": f"cell-{len(cells):02d}", "metadata": {},
                  "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "id": f"cell-{len(cells):02d}", "metadata": {},
                  "execution_count": None, "outputs": [],
                  "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True)})


# =========================================================================== 0. 설정
md('''
# 실차 IPM 실습: rosbag 영상으로 라벨링, 학습, 차에서 달리기까지

**이 노트북의 목표**: 실차 카메라 영상으로 모델을 학습해서, 차가 스스로 두 테이프 사이를 달리게 하는 것.

2주차 시뮬에서 학습한 모델은 **위에서 내려다본 바닥 그림(BEV)** 을 보고 "1 m 앞에서 가야 할 점"을 찍었음.
차는 그 점을 향해 핸들을 꺾음. 그런데 실차 카메라는 앞을 비스듬히 봄. 그래서 할 일은 크게 둘:

1. 실차 영상을 시뮬과 **똑같은 모양의 BEV** 로 바꾸기 (1~4장). 앞을 보는 영상을 바닥 지도로 펴는 일이라 **IPM** 이라고 부름
2. 그 BEV 에 **정답**(가야 할 점)을 달아서 학습하고, 차에서 돌리기 (5~9장)

```
bag 영상 → 렌즈 왜곡 펴기(2장) → 카메라 높이·기울기 재기(3장) → BEV 로 펴기(4장)
        → 정답 달기(5장) → 데이터셋·학습(6~7장) → 차에서 주행(9장)
```

| 장 | 하는 일 | 왜 |
|---|---|---|
| 1 | bag 안에 뭐가 있는지 봄 | 무엇이 있고 무엇이 **없는지** 확인. 없는 것은 2·3장에서 채움 |
| 2 | 색 복원, 렌즈 왜곡 펴기 | 휜 영상으로는 테이프를 직선으로 믿을 수 없음 |
| 3 | 테이프에 점 4개 찍어서 카메라 높이·기울기 재기 (손으로 IPM) | BEV 로 펴려면 카메라가 어디서 어떻게 보는지 알아야 함 |
| 4 | 실차 BEV 와 시뮬 BEV 비교 | 시뮬에서 배운 모델이 통할 모양인지 확인 |
| 5 | 정답 달기: 자동 라벨 → 직접 찍기(클릭) → 둘 비교 | 실차는 정답을 아무도 모름 |
| 6 | 데이터셋 저장 | 2주차 학습 코드가 그대로 읽는 형식으로 |
| 7 | 실차 데이터로 학습 테스트 | 2주차 학습 코드 그대로. 실차 데이터로도 배우는지 확인 |
| 8 | (선택) camsim_lab 에서 쓰기 | 시뮬에서 학습한 모델이 실차에서 얼마나 틀리는지 잼 |
| 9 | 학습한 모델을 차에서 돌리기 | 실제 주행 |

GPU 도 gym 도 필요 없음 (7장 학습은 GPU 가 있으면 빨라짐). 수업 bag (약 1.3 GB) 은 첫 실행 때 구글 드라이브에서 받음.
3장과 5장의 클릭 화면은 코랩에서만 뜸. 내 차로 녹화한 bag 으로 하려면 맨 아래 "내 bag 으로 하기".
''')

md('''
## 0. 설치와 설정
코랩이면 첫 셀이 레포를 clone 하고 필요한 것만 설치함 (rosbags, gdown 등. 1분 안쪽). 로컬이면 아무것도 안 함.
''')
code('''
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/jhcho-53/HYU_AEL_3week.git"     # 다른 fork 나 브랜치를 쓰려면 이 두 줄만 바꿈
REPO_BRANCH = "main"
if IN_COLAB:
    if not os.path.isdir("/content/f1tenth_gym"):
        subprocess.run(["git", "clone", "-q", "--branch", REPO_BRANCH, REPO_URL, "/content/f1tenth_gym"], check=True)
    subprocess.run(["git", "-C", "/content/f1tenth_gym", "pull", "-q", "--ff-only"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rosbags==0.11.5", "gdown", "opencv-python-headless",
                    "pyyaml", "imageio-ffmpeg", "onnx"], check=True)   # onnx: 9장 model.onnx 내보내기
print("colab" if IN_COLAB else "local")
''')
code('''
import copy, json, shutil, time
from pathlib import Path
import numpy as np, cv2, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Video, display

ROOT = Path("/content/f1tenth_gym") if IN_COLAB else next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "camsim").is_dir())
os.chdir(ROOT)
sys.path[:0] = [str(ROOT)]
from camsim import config, camera, track, render, real, viz, week3_lab

plt.rcParams.update({"figure.dpi": 80, "axes.unicode_minus": False})   # 코랩 기본 폰트에 한글 없음. 그래프 글자는 영어로
os.makedirs("out/real", exist_ok=True)

def show(img_bgr, width=640, title=None):
    if title: print(title)
    display(Image(data=cv2.imencode(".png", img_bgr)[1].tobytes(), width=width))
print("repo:", ROOT)
''')

md('''
### 파라미터
값을 바꿨으면 이 셀부터 다시 실행.

- `ahead_m` 은 **camsim_lab 과 같아야 함.** 라벨 기준(중심선 따라 몇 m 앞)이라 다르면 같은 모델로 비교가 안 됨.
- `LANE_WIDTH_MEASURED`: 3장에서 카메라 높이를 구할 때 씀. 영상만으로는 실제 크기를 알 수 없음
  (작은 트랙을 가까이서 찍은 것과 큰 트랙을 멀리서 찍은 것이 똑같이 보임). 그래서 실제로 잰 길이가 하나 필요함:
  두 테이프의 **가운데와 가운데 사이**를 줄자로 잰 값. 렌즈 높이(바닥 ~ 렌즈 가운데)를 자로 쟀으면 `CAM_HEIGHT_MEASURED` 에 넣음
  (넣으면 그 값을 씀). 둘 다 None 이면 `config.yaml` 의 가정값(0.20 m)을 씀.
- `OFFSET_X_M`: camsim 좌표 원점은 **후륜축**. 카메라가 후륜축보다 앞에 있으면 그 거리. 모르면 0 (camsim 기본값과 같음).
- `BAG_DIR`, `OST_FILE`: bag 폴더와 그 카메라의 렌즈 캘리브레이션(`ost.yaml`, 1주차 결과). 내 bag 이면 맨 아래 "내 bag 으로 하기".
''')
code('''
cfg = config.load()
cfg.waypoints.ahead_m = 1.0        # 라벨 waypoint 의 전방 호길이 (m). camsim_lab 과 같게

BAG_DIR = "out/real_bag"           # metadata.yaml, *.db3 이 있는 폴더. 없으면 아래 드라이브에서 받음
OST_FILE = f"{BAG_DIR}/ost.yaml"   # 그 카메라의 렌즈 캘리브레이션 (K, 왜곡 D)
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1dKVcoT88K1Ky4co-fWjh_0aNSpuK7LMr"

LANE_WIDTH_MEASURED = 0.80         # [m] 두 테이프 중심 사이 간격 (실측)
CAM_HEIGHT_MEASURED = None         # [m] 바닥 ~ 렌즈 중심 (실측). 넣으면 차선 간격보다 우선
OFFSET_X_M = 0.0                   # [m] 후륜축 -> 카메라 전방 거리

SKIP_STATIC = True                 # 출발 전 정지 구간 프레임은 거의 같은 이미지라 데이터셋에서 뺌
CLICK_EVERY_S = 0.5                # 5장에서 직접 찍을 프레임 간격 (초)
H_FILE = "out/real/H_i2g.npy"
DATA_OUT = "out/real_dataset"
''')
code('''
def have_bag():
    return (Path(BAG_DIR) / "metadata.yaml").exists() and Path(OST_FILE).exists() and any(Path(BAG_DIR).glob("*.db3"))

CLASS_BAG_DIR = "out/real_bag"                       # 수업 bag 을 받아 두는 곳
if str(BAG_DIR).startswith("/content/drive"):        # 내 bag 을 드라이브에 뒀으면 연결 (맨 아래 "내 bag 으로 하기")
    from google.colab import drive
    drive.mount("/content/drive")
if not have_bag() and BAG_DIR != CLASS_BAG_DIR:      # 내 bag 경로가 틀렸으면 수업 bag 을 받지 않고 멈춤
    raise FileNotFoundError(f"{BAG_DIR} 에 metadata.yaml 이나 .db3 가 없거나 {OST_FILE} 가 없음. 경로를 확인할 것.")
if not have_bag():
    print("Downloading the bag from Google Drive (~1.3 GB). If it stalls, rerun this cell.")
    try:
        import gdown
        gdown.download_folder(DRIVE_FOLDER_URL, output=BAG_DIR, quiet=False)
    except Exception as e:                           # 여러 명이 한꺼번에 받으면 드라이브가 막을 수 있음 (다운로드 할당량)
        print("download failed:", type(e).__name__, e)
if not have_bag():                                   # 그래도 없으면 레포 안 예제 bag: 같은 주행을 0.5초에 한 장으로 줄인 것
    BAG_DIR, OST_FILE = str(week3_lab.SAMPLE / "run_train2_part1_2hz"), str(week3_lab.SAMPLE / "ost.yaml")
    print("드라이브에서 bag 을 못 받아서 레포 안 예제 bag 으로 진행함 (같은 주행, 0.5초에 한 장).")
print("bag:", BAG_DIR, sorted(p.name for p in Path(BAG_DIR).iterdir()))
''')

# =========================================================================== 1. bag
md('''
## 1. bag 살펴보기
**왜**: 녹화한 bag 에 무엇이 들어 있고 무엇이 **없는지** 확인함. 없는 것은 다음 장들에서 채움.

- 카메라 영상 (`/flir_camera/image_raw`): 있음
- 렌즈 정보 (`camera_info` 의 K): 비어 있음 → 2장에서 1주차 캘리브레이션 결과(`ost.yaml`)를 씀
- 카메라를 차에 어떻게 달았는지 (바닥에서 높이, 숙인 각도): 없음 (`/tf` 토픽이 없음) → 3장에서 영상으로 잼

아래 그래프는 프레임마다 바로 앞 장면과 얼마나 다른지, 즉 차가 움직였는지를 보여 줌 (초록 = 출발 전에 서 있던 구간).
서 있는 구간은 같은 그림이 반복돼서 데이터셋에서 뺌 (`SKIP_STATIC`). 3장에서 점을 안 찍었을 때 쓰는 자동 추정도 이 구간을 씀.
''')
code('''
with real.open_bag(BAG_DIR) as reader:
    for c in reader.connections:
        print(f"{c.topic:<28} {c.msgtype:<28} {c.msgcount:>5}")
    info_conn = next(c for c in reader.connections if c.msgtype.endswith("CameraInfo"))
    info = reader.deserialize(next(reader.messages(connections=[info_conn]))[2], info_conn.msgtype)
print("camera_info K:", list(info.k), " -> empty, so intrinsics come from ost.yaml")

stamps, motion = real.scan_motion(BAG_DIR)
N_FRAMES = len(stamps)
FPS = 1e9 / np.diff(stamps).mean()
MOTION_THRESH = 4.0 if FPS >= 10 else 10.0     # 0.5초에 한 장으로 줄인 bag 은 서 있어도 프레임 사이 차이가 3~8 이라 기준을 올림
STATIC_END = real.static_end(motion, MOTION_THRESH)
t_s = (stamps - stamps[0]) / 1e9
print(f"{N_FRAMES} frames, {FPS:.1f} FPS, static: frame 0 ~ {STATIC_END - 1}")
if STATIC_END / FPS < 1.0:
    print(f"출발 전 정지 구간이 {STATIC_END / FPS:.1f}초({STATIC_END}장)뿐임. 3장에서 점 4개를 찍으면 상관없음 "
          "(점을 안 찍었을 때 쓰는 자동 추정만 이 구간을 씀).")

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(t_s, motion, lw=0.8); ax.axhline(MOTION_THRESH, color="r", ls="--", lw=0.8)
ax.axvspan(0, t_s[min(STATIC_END, N_FRAMES - 1)], color="g", alpha=0.15, label="static")
ax.set(xlabel="time [s]", ylabel="mean |frame diff|", title="Motion"); ax.legend(); plt.show()
''')

# =========================================================================== 2. 디코딩
md('''
## 2. 디코딩과 왜곡 보정
**왜**: 렌즈는 곧은 선을 살짝 휘게 찍음 (화면 가장자리일수록 심함). 휜 채로는 3장에서 테이프를 직선으로 보고 점을 찍을 수 없고,
4장에서 바닥으로 펴도 테이프가 휘어 보임. 그래서 1주차 캘리브레이션 결과(`ost.yaml`)로 먼저 폄.

**하는 일**
1. 색 복원: 카메라는 한 픽셀에 빨강·초록·파랑 중 한 색만 담아 보냄 (모자이크). 이걸 컬러 영상으로 바꿈
2. 왜곡 펴기: `ost.yaml` 로 휜 선을 곧게 폄
3. 크기 맞추기: 시뮬 카메라 크기(640x400)로 줄임. 3장에서 만드는 변환표(`H_i2g.npy`)가 이 크기 기준이라,
   시뮬(camsim_lab)과 차에서 같은 표를 그대로 씀

아래 그림은 왼쪽부터 색을 잘못 복원한 것, 제대로 복원한 원본(아직 휨), 왜곡을 펴고 640x400 으로 줄인 것.

**자세히 (몰라도 됨)**
- **Bayer**: 영상은 `bayer_rggb8` 1채널 모자이크. OpenCV 는 Bayer 패턴 이름을 둘째 행 기준으로 붙여서
  ROS 의 `rggb8` 은 `COLOR_BayerBG2BGR` 로 바꿔야 함 (`cv_bridge` 도 같음). `BayerRG` 를 쓰면 노란 테이프가 하늘색이 됨.
- **ost.yaml**: ROS `camera_calibration` 결과. 들어 있는 건 **렌즈 값(intrinsic) 뿐** (K, 왜곡 D). `rectification_matrix` 와
  `projection_matrix` 의 이동 성분은 스테레오용 칸이라 단안에선 단위행렬 / 0 임. 카메라를 어떻게 달았는지는 없음.
- **camsim 해상도**: 실차 1920x1200 을 정확히 1/3 로 줄이면 비율이 같아서 K 만 같이 1/3 하면 됨.
  camsim 기본 카메라는 화각 90 도 가정인데, 실측 K 로 계산하면 85 도쯤임.
''')
code('''
K_FULL, D, FULL_SIZE = real.load_ost(OST_FILE)
UNDIST = real.undistort_maps(K_FULL, D, FULL_SIZE)
def undistort(img):
    return cv2.remap(img, *UNDIST, interpolation=cv2.INTER_LINEAR)
K_CAM = real.scale_intrinsics(K_FULL, FULL_SIZE, real.camsim_size(cfg))
hfov = np.degrees(2 * np.arctan(FULL_SIZE[0] / 2 / K_FULL[0, 0]))
print(f"full {FULL_SIZE}, camsim {real.camsim_size(cfg)}, D = {np.round(D, 4).tolist()}")
print(f"horizontal FOV: real {hfov:.1f} deg vs config.yaml assumption {cfg.camera.hfov_deg:.1f} deg")
print("K (camsim resolution) =\\n", np.round(K_CAM, 2))

DEMO_IDX = [60, 250, 340, 450]                    # 장면 예시 (기본 bag 556장 기준)
if N_FRAMES <= max(DEMO_IDX):                     # 짧은 bag (0.5초에 한 장으로 줄인 bag 등) 이면 같은 비율 위치로
    DEMO_IDX = sorted({int(i * (N_FRAMES - 1) / 556) for i in DEMO_IDX})
msgs = real.load_images(BAG_DIR, DEMO_IDX)
m = msgs[DEMO_IDX[0]]
mosaic = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)
raw = real.decode_image(m)
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].imshow(cv2.cvtColor(cv2.cvtColor(mosaic, cv2.COLOR_BayerRG2BGR), cv2.COLOR_BGR2RGB)); ax[0].set_title("Wrong: COLOR_BayerRG2BGR")
ax[1].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)); ax[1].set_title("Correct: COLOR_BayerBG2BGR (distorted)")
ax[2].imshow(cv2.cvtColor(real.to_camsim(undistort(raw), cfg), cv2.COLOR_BGR2RGB)); ax[2].set_title("Undistorted, camsim 640x400")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()
''')

# =========================================================================== 3. extrinsic
md('''
## 3. 카메라 높이와 기울기 재기 (손으로 IPM)
**왜 하나**
- 모델은 BEV 위에서 "1 m 앞 점"을 m 단위로 찍음. BEV 는 1 px = 1 cm 인 바닥 지도라서 가능함.
- 카메라 영상은 그렇지 않음. 가까운 바닥은 크게, 먼 바닥은 작게 찍혀서, 같은 1 px 이 가까이서는 몇 mm, 멀리서는 몇 cm 임.
- 그래서 영상을 바닥 지도로 **펴야** 함 (이걸 IPM 이라고 함). 펴려면 카메라가 **바닥에서 몇 cm 위에서**, **몇 도 숙여서** 보는지 알아야 함.
  - 비유: 바닥에 놓인 A4 용지를 폰으로 비스듬히 찍으면 사다리꼴로 보임. 폰을 높이 들수록 작게, 더 비스듬히 볼수록 납작하게 보임.
    거꾸로 폰을 어떻게 들었는지 알면 사다리꼴을 다시 직사각형으로 펼 수 있음.
- **틀리면** (3-4 에서 직접 봄)
  - 기울기가 틀리면: 나란한 두 테이프가 BEV 에서 벌어지거나 모이고, 거리도 틀어짐 → 모델이 찍는 점이 실제 자리와 어긋남
  - 높이가 틀리면: BEV 전체가 커지거나 작아짐 → 모델이 "1 m 앞" 이라고 찍은 점이 실제로는 1 m 가 아님
- 1주차 캘리브레이션(`ost.yaml`)은 **렌즈**만 앎. 카메라를 차에 어떻게 달았는지는 모르고, bag 에도 없음 (1장). 그래서 여기서 잼.

**어떻게 재나** (기찻길 사진을 떠올리면 됨)
- 곧은 길의 두 테이프는 실제로는 평행함. 그런데 사진에서는 멀어질수록 가까워지다가 한 점에서 만남. 이 점이 **소실점**.
- 소실점이 사진의 **어느 높이**에 있는지가 카메라 기울기(pitch)를 알려 줌. 카메라를 아래로 숙일수록 소실점이 위로 올라감.
- 기울기를 알면, 두 테이프 사이가 사진에서 얼마나 넓게 보이는지로 카메라 높이를 알 수 있음.
  카메라가 낮을수록 같은 간격이 더 넓게 보임. 그래서 실제 테이프 간격(0장 `LANE_WIDTH_MEASURED`)을 줄자로 재 둬야 함.
- 좌우로 기울어짐(roll)은 0 으로 봄. 카메라를 똑바로 달았으면 문제없음.

순서: **3-1** 곧은 장면 한 장 고르기 → **3-2** 점 4개 찍기 → **3-3** 결과 확인 → **3-4** 값이 틀리면 어떻게 되나.
결과는 `H_i2g.npy` 로 저장해서 4장부터 씀. "영상의 이 픽셀은 바닥에서 후륜축 앞 몇 m, 왼쪽 몇 m" 를 알려 주는 변환표임.
''')

md('''
### 3-1 곧은 장면 한 장 고르기
아래에 bag 에서 고르게 뽑은 장면들이 프레임 번호와 함께 나옴.
**두 테이프가 다 보이고 곧게 뻗은 장면**의 번호를 셀 맨 위 `IPM_FRAME` 에 넣고 이 셀을 다시 실행.
곧은 장면이어야 하는 이유: "두 테이프가 실제로 평행함" 을 이용해서 재는데, 커브에서는 두 테이프가 평행한 직선이 아님.

- 빨간 제목이 지금 고른 장면
- 차가 서 있거나 천천히 갈 때가 좋음 (빨리 달리면 차체가 앞뒤로 흔들려서 기울기가 조금씩 바뀜)
- 녹화 맨 앞일 필요는 없음. 중간에 곧은 곳을 지나간 장면이면 됨
''')
code('''
IPM_FRAME = 0          # 두 테이프가 다 보이는 곧은 장면의 프레임 번호. 아래 그림에서 골라서 바꿈

if not 0 <= IPM_FRAME < N_FRAMES:
    raise ValueError(f"IPM_FRAME 은 0 ~ {N_FRAMES - 1} 사이여야 함.")
pick = sorted({int(i) for i in np.linspace(0, N_FRAMES - 1, 8)} | {IPM_FRAME})
thumbs = real.load_images(BAG_DIR, pick)
fig, axes = plt.subplots(2, (len(pick) + 1) // 2, figsize=(16, 5), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for ax, i in zip(axes.flat, pick):
    ax.imshow(cv2.cvtColor(real.to_camsim(undistort(real.decode_image(thumbs[i])), cfg), cv2.COLOR_BGR2RGB))
    ax.set_title(f"frame {i}" + ("  (IPM_FRAME)" if i == IPM_FRAME else ""), color="r" if i == IPM_FRAME else "k")
plt.tight_layout(); plt.show()
IPM_IMG = undistort(real.decode_image(thumbs[IPM_FRAME]))     # 왜곡을 편 원본 해상도 영상. 3-2 에서 여기에 점을 찍음
''')

md('''
### 3-2 테이프에 점 4개 찍기
1. 한쪽 테이프의 **가운데**를 클릭 (가까운 곳). 같은 테이프의 가운데를 한 번 더 클릭 (먼 곳)
2. 다른 쪽 테이프도 똑같이 2개
3. 그러면 테이프를 따라 선 두 개가 그려지고, 두 선이 만나는 곳(노란 점 = 소실점)과 지평선(점선)이 나옴.
   **두 선이 테이프 위에 잘 겹치면 저장 (Enter)**. 안 겹치면 점을 끌어서 옮김
4. 저장하면 화면 위에 카메라 기울기와 높이가 나옴. 그다음 **3-3 셀부터 끝까지 다시 실행**
   (4장부터 끝까지 이 값으로 BEV 와 라벨을 만듦. 3-3 만 다시 돌리면 5장 라벨과 카메라 값이 어긋남)

- 한 테이프의 두 점은 서로 멀리 찍을수록 정확함 (가까운 점은 화면 아래쪽, 먼 점은 테이프가 흐려지기 전)
- 키보드가 안 먹으면 그림을 한 번 클릭. Backspace 는 마지막 점 지우기
- 점은 `out/real/<bag 폴더 이름>_ipm.json` 에 저장되고 (드라이브에 둔 내 bag 이면 그 폴더에), 셀을 다시 실행해도 남음.
  `IPM_FRAME` 을 바꾸면 새로 찍음
- **모두 실행**으로 돌렸으면 다른 셀이 다 끝난 뒤에 화면이 뜸. 그때 찍고 저장한 다음 3-3 셀부터 끝까지 다시 실행
''')
code('''
IPM_DIR = Path(BAG_DIR) if str(BAG_DIR).startswith("/content/drive") else Path("out/real")   # 드라이브 bag 이면 옆에 저장
IPM_FILE = IPM_DIR / f"{Path(BAG_DIR).name}_ipm.json"
IPM_LANE_W = LANE_WIDTH_MEASURED if CAM_HEIGHT_MEASURED is None else None   # 화면의 높이도 3-3 과 같은 방법으로
ipm = week3_lab.LaneClicker(IPM_IMG, K_FULL, IPM_LANE_W, IPM_FILE,
                            IPM_FRAME, stamps[IPM_FRAME])
ipm.show()
''')

md('''
### 3-3 결과 확인
3-2 에서 저장한 점으로 기울기와 높이를 계산해서 `H_i2g.npy` 로 저장함.

**확인할 것**: 아래 그림 오른쪽(BEV)에서 두 테이프가 **나란한 두 줄**이면 맞은 것
(차가 차선과 조금 틀어져 서 있었으면 비스듬해도 됨).
위로 갈수록 벌어지거나 모이면 기울기가 틀린 것이니 3-2 에서 점을 다시 찍고 3-3 셀부터 다시 실행.

**출력 읽는 법**
- `pitch`: 카메라가 숙인 각도. + 면 아래로 숙임, - 면 위로 들림
- `높이`: 바닥에서 렌즈 가운데까지
- `<-` 뒤: 그 값을 어디서 가져왔는지

점을 안 찍었으면 (코랩이 아니거나, **모두 실행**으로 돌려서 찍기 전에 이 셀이 돌았으면) **자동 추정**을 씀:
출발 전 정지 구간 영상에서 노란 테이프를 찾아, 위에서 봤을 때 두 테이프가 가장 평행해지는 기울기를 고름.
점을 찍었으면 자동 추정도 옆에 숫자로 보여 줌. 둘이 1도 넘게 다르면 둘 중 하나가 틀린 것.
''')
code('''
V_MARGIN = 15                                        # 지평선에서 이만큼 아래부터 바닥으로 봄 (원본 px)

def auto_pitch():
    """자동 추정: 정지 구간 영상에서 위에서 본 두 테이프가 가장 평행해지는 pitch. (pitch, 테이프 간격 / 높이)"""
    idx = list(range(0, max(STATIC_END, 1), 10 if STATIC_END > 30 else 1))   # 줄인 bag 은 정지 구간이 몇 장뿐이라 전부
    imgs = [undistort(real.decode_image(m)) for m in real.load_images(BAG_DIR, idx).values()]
    pitch = 0.0                                      # 1차: 지평선을 영상 가운데(pitch 0)로 보고, 2차: 1차 지평선 아래만 써서 다시
    for _ in range(2):
        masks = [real.lane_mask(img, real.horizon_row(K_FULL, pitch) + V_MARGIN) for img in imgs]
        pitch, cand, costs, gaps = real.estimate_pitch(masks, K_FULL)
    return pitch, float(np.nanmean(gaps[:, list(cand).index(pitch)]))

try:
    AUTO = auto_pitch()
except ValueError:                                   # 정지 구간에 두 테이프가 다 안 보이면 자동 추정은 못 함
    AUTO = None
if ipm.key["frame"] != IPM_FRAME:
    raise ValueError("3-1 에서 IPM_FRAME 을 바꿨으면 3-2 셀을 다시 실행하고 점을 새로 찍을 것.")
pose = ipm.pose()
if pose is not None:
    PITCH, GAP_H, PITCH_SOURCE = pose["pitch_deg"], pose["gap_h"], f"3-2 에서 찍은 점 (frame {IPM_FRAME})"
elif AUTO is not None:
    (PITCH, GAP_H), PITCH_SOURCE = AUTO, "자동 추정. 3-2 에서 점을 찍고 저장하면 그 값을 씀"
else:
    raise ValueError("자동 추정도 안 됨 (출발 전 정지 구간에 두 테이프가 다 안 보임). "
                     "3-2 에서 점 4개를 찍고 저장한 뒤 이 셀을 다시 실행할 것.")

if CAM_HEIGHT_MEASURED is not None:
    HEIGHT, HEIGHT_SOURCE = CAM_HEIGHT_MEASURED, "자로 잰 높이 (CAM_HEIGHT_MEASURED)"
elif LANE_WIDTH_MEASURED is not None:
    HEIGHT, HEIGHT_SOURCE = LANE_WIDTH_MEASURED / GAP_H, f"테이프 간격 {LANE_WIDTH_MEASURED:.2f} m 로 계산"
else:
    HEIGHT, HEIGHT_SOURCE = cfg.camera.height_m, "가정값 (config.yaml). 테이프 간격이나 렌즈 높이를 재서 0장에 넣을 것"
LANE_W = GAP_H * HEIGHT

cfg.camera.pitch_deg, cfg.camera.height_m, cfg.camera.offset_x_m = PITCH, HEIGHT, OFFSET_X_M
H_FULL = real.ground_homography(K_FULL, PITCH, HEIGHT, OFFSET_X_M)          # 원본 해상도. 라벨링에 씀
H_I2G = real.camsim_h_i2g(K_FULL, FULL_SIZE, cfg, PITCH, HEIGHT, OFFSET_X_M)  # camsim 해상도. 모델 입력에 씀
np.save(H_FILE, H_I2G)
cfg.camera.h_i2g_file = H_FILE
V_MIN = real.horizon_row(K_FULL, PITCH) + V_MARGIN

tilt = "아래로 숙임" if PITCH > 0 else "위로 들림" if PITCH < 0 else "수평"
print(f"카메라 기울기 pitch {PITCH:+.2f}도 ({tilt}) <- {PITCH_SOURCE}")
print(f"카메라 높이 {HEIGHT * 100:.1f} cm <- {HEIGHT_SOURCE}")
if pose is not None and AUTO is not None:
    diff = PITCH - AUTO[0]
    print(f"자동 추정은 pitch {AUTO[0]:+.2f}도, 차이 {diff:+.2f}도: "
          + ("1도 넘게 다름. 아래 BEV 를 보고 점을 다시 찍을지 정할 것" if abs(diff) > 1 else "1도 안쪽이라 괜찮음"))
print("saved", H_FILE)

view = real.to_camsim(IPM_IMG, cfg)                  # 왼쪽: 고른 장면 + 지평선(흰 선), 찍은 점과 테이프 선
s = view.shape[1] / IPM_IMG.shape[1]
v_h = int(round(real.horizon_row(K_FULL, PITCH) * s))
cv2.line(view, (0, v_h), (view.shape[1], v_h), (255, 255, 255), 1, cv2.LINE_AA)
if pose is not None:
    vp = np.array(pose["vanishing_point"]) * s
    for k, col in ((0, (0, 0, 255)), (1, (255, 160, 0))):
        a, b = (np.array(q) * s for q in ipm.points[2 * k:2 * k + 2])
        cv2.line(view, tuple(np.round(a if a[1] > b[1] else b).astype(int)), tuple(np.round(vp).astype(int)), col, 2, cv2.LINE_AA)
        for q in (a, b):
            cv2.circle(view, tuple(np.round(q).astype(int)), 5, col, -1)
bev = real.bev_from_camera(IPM_IMG, H_I2G, cfg)     # 오른쪽: 같은 장면의 BEV
show(viz.side_by_side(view, cv2.resize(bev, None, fx=view.shape[0] / bev.shape[0], fy=view.shape[0] / bev.shape[0])),
     width=1000, title=f"frame {IPM_FRAME}: 찍은 점과 지평선 | BEV (두 테이프가 나란하면 맞음)")
''')

md('''
#### 한 번 더 확인 (3-3): 바닥 거리 눈금을 영상에 그리기
3-3 의 값으로 바닥에 줄자를 그려 본 것. 왼쪽(가정값)과 비교하면 카메라 값을 제대로 재는 게 왜 중요한지 보임.

왼쪽은 `config.yaml` 가정 카메라(화각 90 도, 높이 0.20 m, pitch 0), 오른쪽은 실측 K + 3-3 의 기울기와 높이.
초록 선이 후륜축에서 0.5, 1, 1.5, 2, 3 m, 하늘색이 좌우 0.4 m 간격. **오른쪽 격자가 테이프와 나란해야** 맞는 것.
''')
code('''
def draw_grid(img, H_g2i, xs=(0.5, 1.0, 1.5, 2.0, 3.0), ys=(-0.8, -0.4, 0.0, 0.4, 0.8)):
    out = img.copy()
    far = np.linspace(0.3, 30, 200)
    for y in ys:
        uv = camera.project(H_g2i, np.column_stack([far, np.full_like(far, y)]))
        cv2.polylines(out, [np.round(uv).astype(np.int32)], False, (255, 200, 0), 1, cv2.LINE_AA)
    for x in xs:
        uv = np.round(camera.project(H_g2i, np.array([[x, ys[0]], [x, ys[-1]], [x, 0.0]]))).astype(int)
        cv2.line(out, tuple(uv[0]), tuple(uv[1]), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(out, f"{x:g}m", tuple(uv[2] + [4, -4]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return out

demo_cam = real.to_camsim(undistort(real.decode_image(msgs[DEMO_IDX[0]])), cfg)
H_assumed = camera.build(config.load())[0]          # config.yaml 기본 카메라
H_cam = camera.build(cfg)[0]                        # h_i2g_file 이 들어간 cfg -> 실차 카메라
show(viz.side_by_side(draw_grid(demo_cam, H_assumed), draw_grid(demo_cam, H_cam)), width=1000,
     title="config.yaml assumption (hfov 90, h 0.20, pitch 0) | ost.yaml K + 3-3 pitch, height")
''')

md('''
### 3-4 값이 틀리면 어떻게 되나
3-3 의 값을 일부러 틀리게 넣고, 3-1 에서 고른 장면을 BEV 로 펴 봄. 가운데가 3-3 의 값, 초록 가로선은 후륜축에서 1, 2, 3 m.

- **윗줄 (기울기)**: 2도만 틀려도 나란한 두 테이프가 왼쪽(덜 숙임)은 위로 갈수록 벌어지고, 오른쪽(더 숙임)은 모임.
  거리도 같이 틀어짐: 왼쪽은 모든 게 실제보다 멀게, 오른쪽은 가깝게 펴짐. 모델이 찍는 점이 실제 자리와 어긋남
- **아랫줄 (높이)**: 20% 틀리면 그림 전체가 그만큼 작아지거나 커짐. 테이프 간격도, 테이프까지 거리도 같이 틀림.
  모델이 "1 m 앞" 이라고 찍은 점이 실제로는 1.25 m (x0.8) 나 0.83 m (x1.2) 앞이 됨

그래서 두 가지를 챙겨야 함.
- 기울기: 3-3 에서 BEV 의 두 테이프가 나란한지 눈으로 확인할 수 있음
- 높이: 그림으로는 확인이 안 됨 (커지거나 작아져도 두 테이프는 나란함). 0장의 테이프 간격(`LANE_WIDTH_MEASURED`)을
  줄자로 정확히 재는 수밖에 없음

여기서 넣은 틀린 값은 저장되지 않음 (3-3 의 값이 계속 쓰임).
''')
code('''
def bev_with(pitch, height):
    """IPM_IMG 를 이 pitch, height 로 편 BEV. 초록 가로선 = 후륜축에서 1, 2, 3 m"""
    bev = real.bev_from_camera(IPM_IMG, real.camsim_h_i2g(K_FULL, FULL_SIZE, cfg, pitch, height, OFFSET_X_M), cfg)
    for x in (1.0, 2.0, 3.0):
        (u0, v), (u1, _) = render.bev_pixels(np.array([[x, cfg.bev.y_range_m[1]], [x, cfg.bev.y_range_m[0]]]), cfg)
        cv2.line(bev, (int(u0), int(v)), (int(u1), int(v)), (0, 255, 0), 1)
        cv2.putText(bev, f"{x:g}m", (4, int(v) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return bev

rows = [[(PITCH - 2, HEIGHT, f"pitch {PITCH - 2:+.1f} deg (2 deg less)"), (PITCH, HEIGHT, "3-3 value"),
         (PITCH + 2, HEIGHT, f"pitch {PITCH + 2:+.1f} deg (2 deg more)")],
        [(PITCH, HEIGHT * 0.8, f"height {HEIGHT * 80:.1f} cm (x0.8)"), (PITCH, HEIGHT, "3-3 value"),
         (PITCH, HEIGHT * 1.2, f"height {HEIGHT * 120:.1f} cm (x1.2)")]]
fig, axes = plt.subplots(2, 3, figsize=(12, 9))
for ax_row, row in zip(axes, rows):
    for ax, (p, h, name) in zip(ax_row, row):
        ax.imshow(cv2.cvtColor(bev_with(p, h), cv2.COLOR_BGR2RGB)); ax.set_title(name); ax.axis("off")
plt.suptitle(f"frame {IPM_FRAME}: top = wrong pitch, bottom = wrong height"); plt.tight_layout(); plt.show()
''')

md('''
#### 직접 바꿔 보기 (선택)
슬라이더로 pitch 와 높이를 바꾸면서 모델 입력 BEV 가 어떻게 변하는지 봄. 바꾼 값은 저장되지 않음.
''')
code('''
try:
    from ipywidgets import interact, FloatSlider, SelectionSlider
except ImportError:
    print("ipywidgets not found: pip install ipywidgets")
else:
    cache = {i: undistort(real.decode_image(m)) for i, m in msgs.items()}

    @interact(idx=SelectionSlider(options=sorted(cache), value=DEMO_IDX[0], description="frame"),
              pitch=FloatSlider(value=PITCH, min=-12, max=4, step=0.25, description="pitch [deg]"),
              height=FloatSlider(value=HEIGHT, min=0.08, max=0.40, step=0.005, description="height [m]"))
    def _(idx, pitch, height):
        H = real.camsim_h_i2g(K_FULL, FULL_SIZE, cfg, pitch, height, OFFSET_X_M)
        bev = real.bev_from_camera(cache[idx], H, cfg)
        show(viz.side_by_side(draw_grid(real.to_camsim(cache[idx], cfg), np.linalg.inv(H)), bev), width=900,
             title=f"frame {idx}, pitch {pitch:+.2f} deg, height {height:.3f} m")
''')

# =========================================================================== 4. BEV 비교
md('''
## 4. 실차 BEV vs 시뮬 BEV
**왜**: 2주차 모델은 시뮬 BEV 만 보고 배웠음. 실차 BEV 가 시뮬 BEV 처럼 생겨야 같은 모델(또는 같은 학습 방법)이 실차에서 통함.
왼쪽은 시뮬이 같은 카메라로 곧은 길을 그린 것, 오른쪽은 실차 영상을 3장 값으로 편 것.

**확인할 것**
- 출력 줄의 `agreement` 가 99 % 쯤이면, 카메라가 보는 범위가 시뮬과 실차에서 같다는 뜻 (3장 값이 시뮬 카메라에도 제대로 들어감).
  오른쪽 회색이 카메라가 못 보는 곳. 왼쪽 시뮬은 못 보는 곳도 바닥과 같은 회색이라 눈으로는 안 보임
- 그 안의 그림은 다름. 실차엔 바닥 반사, 조명 얼룩, 벽·의자가 바닥에 펴져서 길게 늘어진 것이 있음
  (IPM 은 모든 게 바닥에 붙어 있다고 보고 펴서, 높이가 있는 물체는 길게 늘어남).
  이 차이가 sim-to-real 갭이고, camsim_lab 3장 증강 과제가 메우려는 것

**자세히 (몰라도 됨)**: 모델 입력은 380x300 BEV (전방 0.2~4 m, 좌우 ±1.5 m, 1 cm/px). 실차는 `real.bev_from_camera`
(보정 -> 640x400 -> `render.ipm_bev`), 시뮬은 `render.render_bev` + 카메라 가시 마스크. 둘 다 같은 `H_i2g` 를 씀.
''')
code('''
trk_cfg = copy.deepcopy(cfg)
trk_cfg.lane.follow_walls, trk_cfg.lane.track_width_m = False, round(LANE_W, 3)   # 실습실 트랙처럼 일정 폭
trk = track.from_csv(cfg.closed_loop.centerline_csv, trk_cfg)
h = np.unwrap(trk.heading)
i_straight = int(np.argmin(np.abs(h[60:] - h[:-60])))                # 3 m 동안 가장 곧은 곳
pose = np.array([*trk.center[i_straight], trk.heading[i_straight]])

vis_mask = render.bev_visibility_mask(H_cam, cfg)
bev_sim = render.render_bev(pose, trk.quads, trk_cfg, vis_mask)
bev_real = real.bev_from_camera(undistort(real.decode_image(msgs[DEMO_IDX[0]])), H_I2G, cfg)
floor = np.all(bev_real == cfg.lane.color_floor, axis=-1)
print(f"BEV {bev_real.shape[1]}x{bev_real.shape[0]}, visible area: sim mask {vis_mask.mean()*100:.0f} %, "
      f"real IPM {(~floor).mean()*100:.0f} %, agreement {(vis_mask == ~floor).mean()*100:.1f} %")
show(viz.side_by_side(bev_sim, bev_real), width=700, title=f"sim BEV (straight section, same camera) | real BEV (frame {DEMO_IDX[0]})")
''')

# =========================================================================== 5. 라벨링
md('''
## 5. 라벨링
**왜**: 모델을 학습하려면 BEV 마다 정답이 있어야 함. 시뮬은 정답이 자동으로 나왔지만, 실차는 아무도 정답을 모름.

**정답** = 두 테이프 한가운데를 따라 1 m 앞에 있는 점 (waypoint). 차가 이 점을 계속 따라가면 차선 가운데로 달림.

그래서 5-1 에서 노란 테이프로 정답을 **자동으로** 찾고, 5-2 에서 사람이 **직접 찍어서** 고치고, 5-3 에서 둘을 비교함.

### 5-1 자동 라벨
아래는 왼쪽부터 전면 영상(초록 = 정답 점), 라벨링 격자(빨강·파랑 = 찾은 테이프, 자홍 = 한가운데 선), 모델 입력 BEV.
라벨링 격자는 자동 라벨이 쓰는 바닥 지도 (전방 0.3~2.5 m). 모델 입력 BEV 와는 따로 만듦.
그다음 셀이 bag 의 모든 프레임에 자동 라벨을 달고, 시간에 따른 정답 점 위치를 그래프로 보여 줌
(빨강 = 정답을 못 찾았거나, 앞뒤 프레임과 비교해 튀어서 뺀 프레임).

**자세히 (몰라도 됨)**: 정답 기준은 camsim `gt.waypoint_ahead` 와 같음:
**중심선 위, 후륜축에 가장 가까운 점에서 호길이 `ahead_m` 앞의 점** (후륜축 기준 x, y).

1. 원본 해상도 영상에서 노란 테이프 마스크 -> 지면 격자로 폄 (전방 0.3~2.5 m). 먼저 펴고 색을 고르면 먼 테이프가 흐려져서 놓침
2. 조각난 마스크를 차선 둘로 묶음. 큰 조각부터 기존 차선의 y=f(x) 에 8 cm 안으로 맞으면 합침
3. 중심선: 한 차선의 각 점에서 다른 차선의 최근접점과의 중점. 같은 x 에서 평균 내면 커브에서 틀림
4. 중심선을 후륜축 쪽으로 늘려 최근접점을 찾고 (코앞은 카메라 사각지대) 호길이 `ahead_m` 지점을 waypoint 로
''')
code('''
G = real.LABEL_GRID
def draw_label(img_undist, lab):
    grid = real.warp_to_grid(img_undist, H_FULL, **G)
    for L, col in zip(lab["lanes"], [(0, 0, 255), (255, 0, 0)]):
        cv2.polylines(grid, [real.ground_to_grid(real.sample_lane(L), **G).astype(np.int32)], False, col, 2)
    if lab["center"] is not None:
        cv2.polylines(grid, [real.ground_to_grid(lab["center"], **G).astype(np.int32)], False, (255, 0, 255), 2)
    if np.isfinite(lab["wp"][0]):
        cv2.circle(grid, tuple(real.ground_to_grid(lab["wp"], **G)[0].astype(int)), 5, (0, 255, 0), -1)
    return grid

for i in DEMO_IDX:
    img = undistort(real.decode_image(msgs[i]))
    lab = real.label_frame(img, H_FULL, LANE_W, cfg.waypoints.ahead_m, V_MIN)
    front = real.to_camsim(img, cfg)
    bev = real.bev_from_camera(img, H_I2G, cfg)
    if np.isfinite(lab["wp"][0]):
        render.draw_points(front, lab["wp"], H_cam); render.draw_points_bev(bev, lab["wp"], cfg)
    grid = cv2.resize(draw_label(img, lab), None, fx=400 / 220, fy=400 / 220)
    show(viz.side_by_side(front, grid, cv2.resize(bev, None, fx=400 / 380, fy=400 / 380)), width=1000,
         title=f"frame {i}: {lab['status']}, wp = {np.round(lab['wp'], 3)} m")
''')
code('''
H_LABELED = H_I2G.copy()                       # 5장 라벨을 만든 카메라 값. 5-2 와 6장이 같은지 확인함
def check_camera():
    if not np.array_equal(H_LABELED, H_I2G):
        raise ValueError("5-1 라벨을 만든 뒤에 3장 카메라 값이 바뀌었음. 3-3 셀부터 끝까지 다시 실행할 것.")

records, WPS = [], []
t0 = time.time()
for i, t, msg in real.iter_images(BAG_DIR):
    lab = real.label_frame(undistort(real.decode_image(msg)), H_FULL, LANE_W, cfg.waypoints.ahead_m, V_MIN)
    records.append(dict(frame=i, stamp_ns=t, status=lab["status"]))
    WPS.append(lab["wp"])
WPS = np.array(WPS)
frames = pd.DataFrame(records)
frames["wp_x"], frames["wp_y"] = WPS[:, 0].round(4), WPS[:, 1].round(4)
frames["jump"] = real.flag_jumps(WPS)
frames["static"] = frames["frame"] < STATIC_END
frames["valid"] = (frames["status"] == "two_lanes") & np.isfinite(WPS[:, 0]) & ~frames["jump"]
frames["use"] = frames["valid"] & ~(frames["static"] & SKIP_STATIC)
print(f"labeled {len(frames)} frames in {time.time() - t0:.0f}s")
print(frames["status"].value_counts().to_string())
print(f"valid {frames['valid'].sum()}, jump {frames['jump'].sum()}, usable for dataset {frames['use'].sum()} (before 5-2 clicks)")

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(t_s, WPS[:, 0], label="wp_x (forward)"); ax.plot(t_s, WPS[:, 1], label="wp_y (+left)")
for n in np.flatnonzero(~frames["valid"]):
    ax.axvspan(t_s[n] - 0.013, t_s[n] + 0.013, color="r", alpha=0.3, lw=0)
ax.axvspan(0, t_s[STATIC_END], color="g", alpha=0.08)
ax.set(xlabel="time [s]", ylabel="[m]", title=f"Waypoint {cfg.waypoints.ahead_m:g} m ahead (red = invalid, green = static)")
ax.legend(); ax.grid(alpha=0.3); plt.show()
''')

md('''
### 5-2 직접 찍기 (클릭)
자동 라벨은 테이프가 안 보이거나 하나만 보이면 틀리거나 포기함. 그래서 사람이 정답을 찍는 단계가 필요함.
`CLICK_EVERY_S`(0.5초)마다 한 장씩 BEV 를 띄움 (정지 구간은 건너뜀). 십자 = 뒷바퀴 축, 점선 원 = 뒷바퀴 축에서 1 m.

1. **원이 좌우 테이프의 가운데와 만나는 곳을 클릭** (점이 원 위에 붙음, 드래그로 옮김)
2. 맞으면 **Enter**(승인) → 다음 프레임. 가운데를 모르겠으면(테이프가 안 보임, 흐림) **X**(제외)
3. **←, →** 로 앞뒤 이동, "다음 미작업" 으로 남은 프레임

키보드가 안 먹으면 화면을 한 번 클릭. 라벨은 누를 때마다 `out/real/<bag 폴더 이름>_clicks.json` 에 저장되고
(드라이브에 둔 내 bag 이면 그 폴더에), 셀을 다시 실행해도 남아 있음 (같은 bag 의 같은 프레임, 같은 카메라 자세일 때).
승인한 점은 6장 데이터셋에서 자동 라벨 대신 쓰이고, 제외한 프레임은 데이터셋에서 빠짐. 다 안 찍어도 다음 장으로 넘어갈 수 있음.

**모두 실행**으로 돌렸으면 다른 셀이 도는 동안은 클릭이 안 먹음. 끝까지 다 돈 뒤에 찍고, **6장부터 다시 실행**할 것.
3-2 에서 점을 새로 저장했으면 **먼저 3-3 셀부터 끝까지 다시 실행**한 뒤에 여기서 찍을 것
(3장 값이 바뀌면 BEV 가 바뀌어서 전에 찍은 라벨이 안 맞음).
''')
code('''
check_camera()
start = STATIC_END if SKIP_STATIC else 0                 # 정지 구간은 거의 같은 그림이라 건너뜀
step = max(1, int(round(FPS * CLICK_EVERY_S)))           # 37 Hz bag 이면 19장에 한 장
CLICK_IDX = list(range(start, N_FRAMES, step))
click_msgs = real.load_images(BAG_DIR, CLICK_IDX)
click_bevs = [real.bev_from_camera(undistort(real.decode_image(click_msgs[i])), H_I2G, cfg) for i in CLICK_IDX]
del click_msgs
CLICK_DIR = Path(BAG_DIR) if str(BAG_DIR).startswith("/content/drive") else Path("out/real")   # 드라이브 bag 이면 옆에 저장
CLICK_FILE = CLICK_DIR / f"{Path(BAG_DIR).name}_clicks.json"                                   # bag 마다 따로
labeler = week3_lab.Labeler([dict(stamp_ns=int(stamps[i])) for i in CLICK_IDX], click_bevs, cfg, CLICK_FILE)
print(f"클릭할 프레임 {len(CLICK_IDX)}장 ({step}장에 한 장, {step / FPS:.2f}초 간격)")
labeler.show()
''')
code('''
labeler.counts()      # 다 찍었으면 실행해서 확인
''')

md('''
### 5-3 자동 라벨과 비교
같은 프레임에서 5-1 의 자동 라벨과 내가 찍은 점을 비교함.

- 자동 라벨이 없는 프레임: 차선 둘을 못 찾았거나 앞뒤와 비교해 튀어서 뺀 것. 사람은 찍을 수 있었는지 볼 것
- 둘 다 있는 프레임의 차이가 몇 cm 인지. 정의가 조금 달라서(자동 = 중심선 **따라** 1 m, 클릭 = 1 m **원** 위) 몇 cm 는 정상.
  크게 다르면 어느 쪽이 맞는지 BEV 를 보고 판단
''')
code('''
auto = np.where(frames["valid"].to_numpy()[CLICK_IDX, None], WPS[CLICK_IDX], np.nan)   # 5-1 이 쓸 수 있다고 본 자동 라벨
mine = np.array([l["waypoint_m"] if l["status"] == "accepted" else [np.nan, np.nan] for l in labeler.labels],
                float).reshape(-1, 2)
has_auto, has_mine = np.isfinite(auto).all(1), np.isfinite(mine).all(1)
both = has_auto & has_mine
gap_cm = np.where(both, np.hypot(*(auto - mine).T) * 100, np.nan)
print(f"자동 라벨 {has_auto.sum()} / {len(CLICK_IDX)}장 | 내가 승인 {has_mine.sum()}장 | 둘 다 있음 {both.sum()}장")
print("자동 라벨이 없는 프레임:", [CLICK_IDX[k] for k in np.flatnonzero(~has_auto)])
if both.any():
    print(f"둘의 차이: 평균 {np.nanmean(gap_cm):.1f} cm, 최대 {np.nanmax(gap_cm):.1f} cm")
worst = [int(k) for k in np.argsort(-np.nan_to_num(gap_cm, nan=-1))[:4] if both[k]]   # 차이가 큰 순
if worst:
    fig, axes = plt.subplots(1, len(worst), figsize=(12, 4.5), squeeze=False)
    for ax, k in zip(axes[0], worst):
        img = click_bevs[k].copy()
        render.draw_points_bev(img, mine[k], cfg, (0, 200, 0), 7)      # 초록: 내가 찍은 점
        render.draw_points_bev(img, auto[k], cfg, (255, 128, 0), 5)    # 파랑: 자동 라벨
        ax.imshow(img[..., ::-1]); ax.set_title(f"frame {CLICK_IDX[k]}: {gap_cm[k]:.0f} cm apart"); ax.axis("off")
    plt.suptitle("green: my click   blue: auto label"); plt.show()
''')

# =========================================================================== 6. 저장
md('''
## 6. 데이터셋 저장 (camsim 포맷)
**왜**: 2주차 학습 코드가 그대로 읽는 형식으로 저장함. 그래야 학습 코드를 안 고치고 7장에서 바로 씀.

5-1 의 자동 라벨에 5-2 에서 찍은 것을 덮어씀: **승인한 프레임은 내 점**, 제외한 프레임은 뺌, 나머지는 자동 라벨 그대로.
내 점은 1 m 원 위, 자동 라벨은 중심선을 따라 1 m 라서 몇 cm 다를 수 있음 (5-3).
```
out/real_dataset/
├── images/NNNNNN.png     실차 BEV 380x300 (파일명 = bag 프레임 번호)
├── labels.csv            file, x, y, theta, wp_x, wp_y   <- DiskDataset 이 읽는 포맷 그대로
├── frames.csv            전체 프레임의 상태·플래그 (valid, jump, static, label = auto / click / rejected)
└── real_spec.json        bag, K, D, pitch, 높이, H_i2g 등 만든 설정
```
- 실차엔 world pose 가 없어서 `labels.csv` 의 x, y, theta 는 nan. 학습엔 wp 만 쓰므로 상관없음.
- `spec.json` 은 일부러 안 씀. 그건 시뮬 데이터의 재생성 판단용이라 실차 데이터엔 해당 없음.
- 연속 프레임은 거의 같은 그림이라 `DiskDataset` 의 무작위 9:1 분할은 실차 데이터엔 안 맞음.
  평가는 `split="all"` 로, 학습에 섞을 거면 시간 구간(또는 bag) 단위로 나눌 것.
''')
code('''
check_camera()
WPS_DS = WPS.copy()                                 # 데이터셋에 쓸 라벨: 자동 라벨에 5-2 클릭을 덮어씀
frames["label"] = "auto"
for i, l in zip(CLICK_IDX, labeler.labels):
    if l["status"] == "accepted":
        WPS_DS[i] = l["waypoint_m"]
        frames.loc[i, "label"] = "click"
    elif l["status"] == "rejected":
        frames.loc[i, "label"] = "rejected"
frames["wp_x"], frames["wp_y"] = WPS_DS[:, 0].round(4), WPS_DS[:, 1].round(4)
frames["use"] = ((frames["valid"] | (frames["label"] == "click")) & (frames["label"] != "rejected")
                 & ~(frames["static"] & SKIP_STATIC))
print("label of the frames used:", frames.loc[frames["use"], "label"].value_counts().to_dict())

if Path(DATA_OUT).exists():
    shutil.rmtree(DATA_OUT)                         # 다시 실행하면 새로 만듦
use = set(frames.loc[frames["use"], "frame"])
files = {}
with real.LabelWriter(DATA_OUT) as w:
    for i, t, msg in real.iter_images(BAG_DIR):
        if i in use:
            files[i] = w.add(real.bev_from_camera(undistort(real.decode_image(msg)), H_I2G, cfg), WPS_DS[i], f"{i:06d}.png")
frames["file"] = frames["frame"].map(files)
frames.to_csv(f"{DATA_OUT}/frames.csv", index=False)
spec = {"bag": str(BAG_DIR), "n": len(files), "K": K_FULL.tolist(), "D": D.tolist(), "full_size": list(FULL_SIZE),
        "pitch_deg": PITCH, "pitch_source": PITCH_SOURCE, "height_m": HEIGHT, "height_source": HEIGHT_SOURCE,
        "offset_x_m": OFFSET_X_M, "lane_width_m": LANE_W, "ahead_m": cfg.waypoints.ahead_m, "bev": vars(cfg.bev),
        "H_i2g": H_I2G.tolist()}
Path(f"{DATA_OUT}/real_spec.json").write_text(json.dumps(spec, indent=1))
size_mb = sum(p.stat().st_size for p in Path(DATA_OUT).rglob("*") if p.is_file()) / 1e6
print(f"saved {len(files)} BEV images + labels.csv -> {DATA_OUT} ({size_mb:.0f} MB)")
if len(files) < 5:
    raise ValueError(f"데이터셋이 {len(files)}장뿐임. 7장에서 20% 를 검증용으로 떼고 학습하려면 5장 이상 필요함. "
                     "5-2 에서 제외(X)를 줄이거나 더 긴 bag 을 쓸 것.")
''')

md('''
### 저장된 걸 학습 코드처럼 읽어 보기
camsim 학습 코드가 쓰는 `dataset.DiskDataset` 으로 읽음 (torch 필요. 코랩엔 있음).
''')
code('''
try:
    from camsim import dataset
except ImportError:
    dataset = None
    print("torch not installed: checking labels.csv with pandas only")
lab_df = pd.read_csv(f"{DATA_OUT}/labels.csv")
display(lab_df.head(3))
if dataset is not None:
    ds_real = dataset.DiskDataset(DATA_OUT, cfg, split="all")
    x, y = ds_real[0]
    print(f"DiskDataset: {len(ds_real)} samples, input {tuple(x.shape)}, target {y.numpy().round(3)} (x, y) / norm_m")

pick = lab_df.sample(min(4, len(lab_df)), random_state=0).sort_values("file")
tiles = []
for _, r in pick.iterrows():
    bev = cv2.imread(f"{DATA_OUT}/images/{r.file}")
    tiles.append(render.draw_points_bev(bev, [r.wp_x, r.wp_y], cfg))
show(viz.side_by_side(*tiles), width=1000, title="  |  ".join(pick["file"]))
''')
code('''
# 검수용 영상: 전면 (camsim 해상도) | 모델 입력 BEV, 초록 = 라벨. 빨간 글씨 = 데이터셋에서 뺀 프레임
video_path = "out/real/labels_preview.mp4"
writer = None
for i, t, msg in real.iter_images(BAG_DIR):
    img = undistort(real.decode_image(msg))
    front, bev = real.to_camsim(img, cfg), real.bev_from_camera(img, H_I2G, cfg)
    if np.isfinite(WPS_DS[i, 0]):
        render.draw_points(front, WPS_DS[i], H_cam); render.draw_points_bev(bev, WPS_DS[i], cfg)
    panel = viz.side_by_side(front, cv2.resize(bev, None, fx=400 / 380, fy=400 / 380))
    r = frames.loc[i]
    cv2.putText(panel, f"frame {i} {r.status} {r.label} {'used' if r.use else 'skipped'}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if r.use else (0, 0, 255), 2)
    if writer is None:
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 1e9 / np.diff(stamps).mean(),
                                 (panel.shape[1], panel.shape[0]))
    writer.write(panel)
writer.release()
print("saved", video_path)
try:
    display(Video(viz.to_h264(video_path), embed=True, width=900))   # mp4v 는 브라우저가 못 읽어서 H.264 로
except Exception as e:
    print("inline playback unavailable:", type(e).__name__)
''')

# =========================================================================== 7. 학습 테스트
md('''
## 7. 실차 데이터로 학습 테스트
6장에서 저장한 실차 데이터를 camsim 학습 코드(`train.train`)에 그대로 넣어 봄.

- **분할**: `DiskDataset` 의 기본 분할은 무작위 9:1 인데, 연속 프레임은 거의 같은 그림이라 val 에 train 과 같은 장면이 들어가 점수가 부풀려짐.
  그래서 **시간순**으로 앞쪽을 train, 마지막 `VAL_FRAC` 를 val 로 두고 경계에 `GAP` 프레임을 버림.
- **모델**: 기본은 `small` (작은 CNN). CPU 에서도 몇 분이면 끝남. GPU 런타임이면 `resnet18` 로 바꿔 볼 것 (ImageNet 가중치를 받음).
- **기준선**: train 라벨 평균을 항상 내는 "상수 예측". 모델이 이것보다 확실히 나아야 뭔가 배운 것.

train 오차는 작은데 val 오차만 크면 외운 것. val 구간(주행 후반)은 커브가 많아 train 과 장면이 달라서 그 차이도 드러남.
''')
code('''
TRAIN_ARCH = "small"     # "small" | "resnet18" (GPU 권장)
TRAIN_STEPS = 300
TRAIN_BATCH = 16
TRAIN_LR = 1e-3          # resnet18 이면 3e-4 (사전학습 가중치가 초반에 망가지지 않게)
VAL_FRAC = 0.2           # 시간순 마지막 20 % 를 val
GAP = int(round(FPS * 0.3))   # train/val 경계에서 버릴 프레임 수 (0.3 s 어치). 경계 양쪽이 거의 같은 그림이라

if dataset is None:
    print("torch not installed: skipped (Colab has torch)")
else:
    import torch
    from torch.utils.data import Subset
    from camsim import model, train

    train_cfg = copy.deepcopy(cfg)
    train_cfg.model.arch = TRAIN_ARCH
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ds_all = dataset.DiskDataset(DATA_OUT, train_cfg, split="all")
    frame_no = np.array([int(f.split(".")[0]) for f in ds_all.files])   # 파일명 = bag 프레임 번호
    order = np.argsort(frame_no)                                         # 시간순
    n_val = max(1, int(round(len(order) * VAL_FRAC)))
    va_idx = order[len(order) - n_val:]
    tr_idx = order[frame_no[order] < frame_no[va_idx[0]] - GAP]          # val 바로 앞 GAP 프레임(0.3 초)은 버림
    if len(tr_idx) == 0:
        raise ValueError("train 으로 쓸 프레임이 없음. 6장 데이터셋이 너무 적음.")
    ds_tr, ds_va = Subset(ds_all, tr_idx.tolist()), Subset(ds_all, va_idx.tolist())
    print(f"train {len(ds_tr)} (frames {ds_all.files[tr_idx[0]]}..{ds_all.files[tr_idx[-1]]}), "
          f"val {len(ds_va)} (frames {ds_all.files[va_idx[0]]}..{ds_all.files[va_idx[-1]]}), device {DEVICE}")

    batch = min(TRAIN_BATCH, len(ds_tr))                   # 짧은 bag 이면 train 이 배치보다 적을 수 있음
    t0 = time.time()
    net, hist = train.train(None, train_cfg, steps=TRAIN_STEPS, batch_size=batch, lr=TRAIN_LR, device=DEVICE,
                            out_path="out/real/model_real_test.pt", dataset=ds_tr, val_dataset=ds_va,
                            val_batches=len(ds_va) // batch + 1, log_every=max(TRAIN_STEPS // 10, 1))
    print(f"trained {TRAIN_STEPS} steps in {time.time() - t0:.0f}s -> out/real/model_real_test.pt")

    st = [h["step"] for h in hist]
    plt.figure(figsize=(7, 3))
    plt.plot(st, [h["loss"] for h in hist], label="train"); plt.plot(st, [h["val_loss"] for h in hist], label="val (time split)")
    plt.yscale("log"); plt.xlabel("step"); plt.ylabel("Huber loss"); plt.grid(alpha=0.3); plt.legend(); plt.show()
''')
code('''
if dataset is not None:
    pred = model.Predictor(net, train_cfg, DEVICE)
    def errors(idx):
        p = np.array([pred.predict(ds_all.load_image(int(k))) for k in idx])
        return p, np.linalg.norm(p - ds_all.wps[idx], axis=1)
    p_tr, e_tr = errors(tr_idx)
    p_va, e_va = errors(va_idx)
    e_const = np.linalg.norm(ds_all.wps[va_idx] - ds_all.wps[tr_idx].mean(0), axis=1)
    display(pd.DataFrame({"mean (cm)": [e_tr.mean() * 100, e_va.mean() * 100, e_const.mean() * 100],
                          "max (cm)": [e_tr.max() * 100, e_va.max() * 100, e_const.max() * 100]},
                         index=["model on train", "model on val", "constant baseline on val"]).round(1))

    frame_no = np.array([int(f.split(".")[0]) for f in ds_all.files])
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.2), sharex=True)
    for a, c, name in ((ax[0], 0, "wp_x (forward)"), (ax[1], 1, "wp_y (+left)")):
        a.plot(frame_no[order], ds_all.wps[order, c], "k", lw=1, label="label")
        a.plot(frame_no[tr_idx], p_tr[:, c], ".", ms=2, label="pred (train)")
        a.plot(frame_no[va_idx], p_va[:, c], ".", ms=2, label="pred (val)")
        a.set(xlabel="bag frame", ylabel="[m]", title=name); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8); plt.tight_layout(); plt.show()

    tiles = []
    for k in va_idx[np.linspace(0, len(va_idx) - 1, 4).astype(int)]:
        bev = ds_all.load_image(int(k))
        render.draw_points_bev(bev, ds_all.wps[k], cfg, (0, 255, 0))
        render.draw_points_bev(bev, pred.predict(ds_all.load_image(int(k))), cfg, (255, 0, 255))
        tiles.append(bev)
    show(viz.side_by_side(*tiles), width=1000, title="val frames: green = label, magenta = model")
''')

# =========================================================================== 8. camsim_lab
md('''
## 8. camsim_lab 에서 쓰기 (선택)
시뮬에서 학습한 모델이 실차에서 얼마나 틀리는지 재 보거나, 실차 데이터를 시뮬 학습에 섞고 싶을 때 함. 9장으로 바로 가도 됨.

코랩은 노트북마다 런타임이 따로라 파일이 안 넘어감. 아래 셀이 드라이브 `MyDrive/camsim_results/real/` 로 옮김
(`H_i2g.npy`, `real_spec.json`, `real_dataset.zip`).

camsim_lab 에선
1. 첫 셀(설치) 다음에 드라이브를 마운트하고 `H_i2g.npy` 와 `real_dataset.zip` 을 `out/` 으로 복사 (`unzip`)
2. 파라미터 셀에 아래 출력의 줄들을 붙임. 시뮬 카메라가 실차 카메라와 같아지고, 테이프도 실습실 트랙처럼 일정 폭이 됨
3. 3장 학습 뒤 실차 오차:
   `ds_real = dataset.DiskDataset("out/real_dataset", cfg, "all")`, `train.evaluate_dataset(pred, ds_real)`
''')
code('''
print("# camsim_lab parameter cell")
print(f'cfg.camera.h_i2g_file = "{H_FILE}"')
print(f"cfg.camera.height_m = {HEIGHT:.4f}     # jitter_bev uses these as the base pose")
print(f"cfg.camera.pitch_deg = {PITCH:.2f}")
print(f"cfg.camera.offset_x_m = {OFFSET_X_M}")
print("cfg.lane.follow_walls = False")
print(f"cfg.lane.track_width_m = {LANE_W:.2f}")
print(f"cfg.waypoints.ahead_m = {cfg.waypoints.ahead_m}")

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    dst = Path("/content/drive/MyDrive/camsim_results/real")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(H_FILE, dst / "H_i2g.npy")
    shutil.copy2(f"{DATA_OUT}/real_spec.json", dst / "real_spec.json")
    archive = shutil.make_archive("out/real_dataset", "zip", root_dir="out", base_dir="real_dataset")
    shutil.copy2(archive, dst / "real_dataset.zip")      # 드라이브는 작은 파일 많으면 느려서 zip 하나로
    print("copied to", dst)
''')

md('''
### 학습된 모델로 실차 오차 재기 (선택)
camsim_lab 5장이 드라이브에 올린 `model.pt` 경로를 `MODEL_PT` 에 넣으면, 실차 BEV 에서 예측한 waypoint 와 6장 라벨의 차이를 잼.
체크포인트의 BEV·waypoint·테이프 색·모델 구조 설정이 이 노트북의 `cfg` 와 다르면 `model.load` 가 거부함
(camsim_lab 에서 바꾼 값이 있으면 여기 파라미터 셀에도 똑같이 넣을 것).

시뮬에서만 학습한 모델이면 오차가 꽤 클 것. 그게 sim-to-real 갭이고, 증강 과제 전후로 이 숫자를 비교하면 됨.
''')
code('''
MODEL_PT = None      # 예: "/content/drive/MyDrive/camsim_results/model.pt"

if MODEL_PT is None:
    print("MODEL_PT not set: skipped")
else:
    import torch
    from camsim import model, train
    pred = model.Predictor(model.load(MODEL_PT, cfg), cfg, "cuda" if torch.cuda.is_available() else "cpu")
    ds_real = dataset.DiskDataset(DATA_OUT, cfg, split="all")
    res = train.evaluate_dataset(pred, ds_real)
    print(f"real data ({res['n']} frames): mean {res['mean_m']*100:.1f} cm, max {res['max_m']*100:.1f} cm")
    plt.figure(figsize=(7, 3)); plt.hist(res["errs_m"] * 100, 30)
    plt.axvline(cfg.lane.tape_width_m * 100, color="r", ls="--", label="tape width")
    plt.xlabel("waypoint error on real BEV (cm)"); plt.ylabel("count"); plt.legend(); plt.show()
    worst = np.argsort(res["errs_m"])[-4:]
    tiles = []
    for k in worst:
        bev = ds_real.load_image(int(k))
        render.draw_points_bev(bev, ds_real.wps[ds_real.idx[k]], cfg, (0, 255, 0))
        render.draw_points_bev(bev, pred.predict(ds_real.load_image(int(k))), cfg, (255, 0, 255))
        tiles.append(bev)
    show(viz.side_by_side(*tiles), width=1000, title="worst 4: green = label, magenta = model")
''')

# =========================================================================== 9. 차
md('''
## 9. 학습한 모델을 차에서 돌리기
7장에서 학습한 모델을 차에 옮겨서 돌림.
차에서는 카메라 영상이 한 장 들어올 때마다 모델이 1 m 앞에서 가야 할 점(waypoint)을 찍고, 그 좌표를 `/waypoint` 토픽으로 보냄.
주행 노드가 그 점을 받아서 그 점으로 가는 조향각을 계산하고 (2주차 시뮬레이터와 같은 pure pursuit), 조이스틱 RB 를 누르고 있는 동안 차를 달리게 함.

아래 셀을 실행하면 차에 가져갈 파일 4개를 `out/car_model/` 폴더에 모음.
코랩이면 이 폴더를 `car_model.zip` 으로 묶어서 구글 드라이브의 **내 드라이브 → `camsim_results` → `car_model.zip`** 에도 올림 (다시 실행하면 덮어씀).
차에서는 레포의 `ros2/README.md` 를 순서대로 따라 하면 됨. 0-2 에서 이 zip 을 차로 받아서 풀고, A 에서 bag 으로 확인하고, B 에서 트랙을 달림.

zip 안의 파일 4개:
- `model.onnx` : 7장에서 학습한 모델
- `checkpoint.json` : 모델 설정. 차에서 모델 파일이 맞는지 확인할 때 씀
- `H_i2g.npy` : 3장에서 잰 카메라 기울기·높이로 만든 BEV 변환. 차에서도 학습 때와 똑같은 BEV 를 만듦
- `ost.yaml` : 렌즈 왜곡을 펴는 값 (2장에서 쓴 것)

차 카메라는 이 bag 을 녹화할 때와 같은 자리, 같은 각도여야 함. 카메라를 건드리면 BEV 가 틀어져서 모델이 엉뚱한 점을 찍음.
''')
code('''
if dataset is None:
    print("torch not installed: skipped (Colab has torch)")
else:
    from camsim import handoff
    CAR_DIR = Path("out/car_model")
    if CAR_DIR.exists():
        shutil.rmtree(CAR_DIR)
    onnx_path = model.export_onnx(net, train_cfg, "out/real/model.onnx")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    handoff.export_checkpoint(Path(onnx_path), CAR_DIR, train_cfg, commit)
    shutil.copy2(H_FILE, CAR_DIR / "H_i2g.npy")
    shutil.copy2(OST_FILE, CAR_DIR / "ost.yaml")
    print("saved", CAR_DIR, sorted(p.name for p in CAR_DIR.iterdir()))
    if IN_COLAB:
        from google.colab import drive
        drive.mount("/content/drive")
        dst = Path("/content/drive/MyDrive/camsim_results")
        dst.mkdir(parents=True, exist_ok=True)
        archive = shutil.make_archive("out/car_model", "zip", root_dir="out", base_dir="car_model")
        shutil.copy2(archive, dst / "car_model.zip")
        print("copied to", dst / "car_model.zip")
''')

# =========================================================================== 정리
md('''
## 정리
- 녹화(rosbag) → 디코딩·왜곡 보정 → 카메라 자세 추정 → BEV (IPM) → 자동 라벨 + 사람이 찍은 라벨 → camsim 형식 데이터셋 → 같은 코드로 학습
- 자동 라벨과 사람 라벨이 언제 갈리는지, 자동이 못 한 프레임은 어떤 장면이었는지 볼 것
- 400장이 아니라 4000장이 필요하면 어떤 장면을 더 녹화해야 할지 생각해 볼 것 (조명, 코너, 반대 방향, 다른 트랙)
''')
md('''
## 내 bag 으로 하기
1. 차에서 레포의 `rosbag/README.md` 순서대로 녹화함. 곧은 구간에서 두 테이프가 다 보이는 장면이 한 번은 있어야 함
   (3장에서 그 장면에 점을 찍음). 테이프 간격(가운데 ~ 가운데)도 줄자로 재 둠.
   파일이 크면 0.5초에 한 장으로 줄여도 됨 (`rosbag/README.md` 5번). 줄인 bag 도 이 노트북이 그대로 읽음
2. bag 폴더(`metadata.yaml` + `.db3`)에 그 카메라의 `ost.yaml` 을 같이 넣어서 노트북 컴퓨터로 가져오고(scp 나 USB),
   **구글 드라이브**의 `내 드라이브/week3/` 에 올림. 런타임이 끊겨도 남고, 다음에 또 쓸 수 있음
3. 0장 파라미터 셀의 `BAG_DIR` 을 아래처럼 바꾸고, 잰 테이프 간격을 `LANE_WIDTH_MEASURED` 에 넣어서 그 셀부터 다시 실행.
   드라이브 경로면 다음 셀이 드라이브를 연결함 (처음 한 번 구글 계정 허용 창이 뜸)
4. 3장에서 내 bag 의 곧은 장면을 골라 점 4개를 새로 찍음 (bag 이 바뀌면 점도 새로 찍어야 함)

```python
BAG_DIR = "/content/drive/MyDrive/week3/my_run_2hz"     # metadata.yaml, .db3, ost.yaml 이 있는 폴더
```
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "real_ipm_lab.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
print("wrote", os.path.normpath(out), len(cells), "cells")
