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
# 실차 IPM 실습 — bag 에서 camsim 규격 데이터까지

`camsim_lab.ipynb` 는 시뮬로 BEV 를 그려서 학습함. 이 노트북은 **실차 카메라 bag** 으로 같은 규격을 만듦.

- `out/real/H_i2g.npy` : 실차 카메라 homography. camsim_lab 파라미터 셀의 `cfg.camera.h_i2g_file` 로 넣으면 시뮬 카메라가 실차와 같아짐
- `out/real_dataset/` : 실차 BEV + waypoint 라벨 (`labels.csv`). `dataset.DiskDataset` 이 그대로 읽음.
  시뮬에서 학습한 모델의 **실차 오차**를 재거나, 실차 데이터로 추가 학습할 때 씀

GPU 도 gym 도 필요 없음 (7장 학습 테스트는 GPU 가 있으면 빨라짐). 실차 bag (약 1.3 GB) 은 첫 실행 때 구글 드라이브에서 받음.

| 장 | 내용 |
|---|---|
| 1 | bag 살펴보기: 토픽, FPS, 정지 구간 |
| 2 | 디코딩과 왜곡 보정: Bayer, `ost.yaml`, camsim 해상도 |
| 3 | 바닥 기준 extrinsic: pitch 추정, 높이, `H_i2g.npy` |
| 4 | 실차 BEV vs 시뮬 BEV |
| 5 | waypoint 자동 라벨링 |
| 6 | 데이터셋 저장 (camsim 포맷) |
| 7 | 실차 데이터로 학습 테스트 (camsim 학습 코드 그대로) |
| 8 | camsim_lab 에서 쓰기 + 학습된 모델로 실차 오차 재기 |
''')

md('''
## 0. 설치와 설정
코랩이면 첫 셀이 레포를 clone 하고 필요한 것만 설치함 (rosbags, gdown 등. 1분 안쪽). 로컬이면 아무것도 안 함.
''')
code('''
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/jhcho-53/HYU_AEL_3week.git"     # 다른 fork 쓰려면 여기만 바꿈
if IN_COLAB:
    if not os.path.isdir("/content/f1tenth_gym"):
        subprocess.run(["git", "clone", "-q", "--branch", "main", REPO_URL, "/content/f1tenth_gym"], check=True)
    subprocess.run(["git", "-C", "/content/f1tenth_gym", "pull", "-q", "--ff-only"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rosbags", "gdown", "opencv-python-headless",
                    "pyyaml", "imageio-ffmpeg"], check=True)
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
from camsim import config, camera, track, render, real, viz

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
- 바닥 기준 높이는 영상만으론 알 수 없어서 실측 하나가 필요함. **차선 간격(테이프 중심 간)** 을 줄자로 재는 게 렌즈 높이보다 쉽고 정확함.
  둘 다 None 이면 `config.yaml` 의 가정값(0.20 m)을 씀.
- `OFFSET_X_M`: camsim 좌표 원점은 **후륜축**. 카메라가 후륜축보다 앞에 있으면 그 거리. 모르면 0 (camsim 기본값과 같음).
''')
code('''
cfg = config.load()
cfg.waypoints.ahead_m = 1.0        # 라벨 waypoint 의 전방 호길이 (m). camsim_lab 과 같게

BAG_DIR = "out/real_bag"           # metadata.yaml, ost.yaml, *.db3 이 있는 폴더. 없으면 아래 드라이브에서 받음
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1dKVcoT88K1Ky4co-fWjh_0aNSpuK7LMr"

LANE_WIDTH_MEASURED = 0.80         # [m] 두 테이프 중심 사이 간격 (실측)
CAM_HEIGHT_MEASURED = None         # [m] 바닥 ~ 렌즈 중심 (실측). 넣으면 차선 간격보다 우선
OFFSET_X_M = 0.0                   # [m] 후륜축 -> 카메라 전방 거리

SKIP_STATIC = True                 # 출발 전 정지 구간 프레임은 거의 같은 이미지라 데이터셋에서 뺌
H_FILE = "out/real/H_i2g.npy"
DATA_OUT = "out/real_dataset"
''')
code('''
DATA_FILES = ["metadata.yaml", "ost.yaml"]
if not (all((Path(BAG_DIR) / f).exists() for f in DATA_FILES) and list(Path(BAG_DIR).glob("*.db3"))):
    import gdown
    print("Downloading the bag from Google Drive (~1.3 GB). If it stalls, rerun this cell.")
    gdown.download_folder(DRIVE_FOLDER_URL, output=BAG_DIR, quiet=False)
print("bag:", sorted(p.name for p in Path(BAG_DIR).iterdir()))
''')

# =========================================================================== 1. bag
md('''
## 1. bag 살펴보기
ROS 설치 없이 `rosbags` 로 읽음. 카메라 토픽 하나만 녹화돼 있고 **`camera_info` 는 0 으로 비어 있음.**
`/tf` 도 없어서 카메라가 바닥에서 얼마나, 어떤 각도로 달렸는지(extrinsic)는 bag 어디에도 없음. 3장에서 영상으로 추정함.

프레임 간 밝기 차로 움직임을 재서, 출발 전 **정지 구간**을 찾음. 3장 pitch 추정은 차선이 곧은 이 구간으로 함.
''')
code('''
with real.open_bag(BAG_DIR) as reader:
    for c in reader.connections:
        print(f"{c.topic:<28} {c.msgtype:<28} {c.msgcount:>5}")
    info_conn = next(c for c in reader.connections if c.msgtype.endswith("CameraInfo"))
    info = reader.deserialize(next(reader.messages(connections=[info_conn]))[2], info_conn.msgtype)
print("camera_info K:", list(info.k), " -> empty, so intrinsics come from ost.yaml")

stamps, motion = real.scan_motion(BAG_DIR)
STATIC_END = real.static_end(motion)
N_FRAMES = len(stamps)
t_s = (stamps - stamps[0]) / 1e9
print(f"{N_FRAMES} frames, {1e9 / np.diff(stamps).mean():.1f} FPS, static: frame 0 ~ {STATIC_END - 1}")

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(t_s, motion, lw=0.8); ax.axhline(4.0, color="r", ls="--", lw=0.8)
ax.axvspan(0, t_s[STATIC_END], color="g", alpha=0.15, label="static")
ax.set(xlabel="time [s]", ylabel="mean |frame diff|", title="Motion"); ax.legend(); plt.show()
''')

# =========================================================================== 2. 디코딩
md('''
## 2. 디코딩과 왜곡 보정
**Bayer**: 영상은 `bayer_rggb8` 1채널 모자이크. OpenCV 는 Bayer 패턴 이름을 둘째 행 기준으로 붙여서
ROS 의 `rggb8` 은 `COLOR_BayerBG2BGR` 로 바꿔야 함 (`cv_bridge` 도 같음). `BayerRG` 를 쓰면 노란 테이프가 하늘색이 됨.

**ost.yaml**: ROS `camera_calibration` 결과. 들어 있는 건 **intrinsic 뿐** (K, 왜곡 D). `rectification_matrix` 와
`projection_matrix` 의 이동 성분은 스테레오용 칸이라 단안에선 단위행렬 / 0 임. 바닥 기준 자세는 없음.

**camsim 해상도**: camsim 카메라는 640x400. 실차 1920x1200 을 정확히 1/3 로 줄이면 비율이 같아서 K 만 같이 1/3 하면 됨.
camsim 기본 카메라는 화각 90 도 가정인데, 실측 K 로 계산하면 85 도쯤임.
''')
code('''
K_FULL, D, FULL_SIZE = real.load_ost(f"{BAG_DIR}/ost.yaml")
UNDIST = real.undistort_maps(K_FULL, D, FULL_SIZE)
def undistort(img):
    return cv2.remap(img, *UNDIST, interpolation=cv2.INTER_LINEAR)
K_CAM = real.scale_intrinsics(K_FULL, FULL_SIZE, real.camsim_size(cfg))
hfov = np.degrees(2 * np.arctan(FULL_SIZE[0] / 2 / K_FULL[0, 0]))
print(f"full {FULL_SIZE}, camsim {real.camsim_size(cfg)}, D = {np.round(D, 4).tolist()}")
print(f"horizontal FOV: real {hfov:.1f} deg vs config.yaml assumption {cfg.camera.hfov_deg:.1f} deg")
print("K (camsim resolution) =\\n", np.round(K_CAM, 2))

DEMO_IDX = [60, 250, 340, 450]
msgs = real.load_images(BAG_DIR, DEMO_IDX)
m = msgs[60]
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
## 3. 바닥 기준 extrinsic
IPM 에는 카메라가 바닥에서 **얼마나 높이, 어떤 각도로** 달렸는지가 필요함. bag 에 없으니 영상으로 정함.

- **pitch**: 정지 구간 프레임에서 노란 테이프를 HSV 로 고르고, pitch 를 바꿔 가며 위에서 본 두 차선의 간격이
  행마다 얼마나 일정한지(변동계수 std/mean)를 잼. 가장 일정할 때가 정답. 차선이 곧은 구간에서만 성립함.
- **높이**: 지면이 통째로 높이에 비례해 커지거나 작아질 뿐 영상은 똑같아서, 영상만으론 못 정함. 위에서 차선 간격이
  "높이의 몇 배"로 나오므로 실측 간격 하나로 높이가 정해짐.
- roll, yaw 는 0 으로 가정. 정확히 하려면 바닥에 체커보드를 놓고 `cv2.solvePnP` 로 네 값을 한 번에 구하면 됨.

결과를 `H_i2g.npy` (camsim 해상도 이미지 px -> 후륜축 기준 지면 m) 로 저장하고 `cfg.camera.h_i2g_file` 에 넣음.
''')
code('''
V_MARGIN = 15                                        # 지평선에서 이만큼 아래부터 바닥으로 봄 (원본 px)
static_idx = list(range(0, STATIC_END, 10))
static_frames = {i: undistort(real.decode_image(m)) for i, m in real.load_images(BAG_DIR, static_idx).items()}
PITCH = 0.0                                          # 1차: 지평선을 영상 중앙(pitch 0)으로 보고 추정
for _ in range(2):                                   # 2차: 1차 pitch 의 지평선 아래만 써서 다시
    masks = [real.lane_mask(static_frames[i], real.horizon_row(K_FULL, PITCH) + V_MARGIN) for i in static_idx]
    PITCH, cand, costs, gaps = real.estimate_pitch(masks, K_FULL)
GAP_H = float(np.nanmean(gaps[:, list(cand).index(PITCH)]))      # 차선 간격 / 카메라 높이

if CAM_HEIGHT_MEASURED is not None:
    HEIGHT, HEIGHT_SOURCE = CAM_HEIGHT_MEASURED, "measured height"
elif LANE_WIDTH_MEASURED is not None:
    HEIGHT, HEIGHT_SOURCE = LANE_WIDTH_MEASURED / GAP_H, "from measured lane width"
else:
    HEIGHT, HEIGHT_SOURCE = cfg.camera.height_m, "ASSUMED (config.yaml)"
LANE_W = GAP_H * HEIGHT

cfg.camera.pitch_deg, cfg.camera.height_m, cfg.camera.offset_x_m = PITCH, HEIGHT, OFFSET_X_M
H_FULL = real.ground_homography(K_FULL, PITCH, HEIGHT, OFFSET_X_M)          # 원본 해상도. 라벨링에 씀
H_I2G = real.camsim_h_i2g(K_FULL, FULL_SIZE, cfg, PITCH, HEIGHT, OFFSET_X_M)  # camsim 해상도. 모델 입력에 씀
np.save(H_FILE, H_I2G)
cfg.camera.h_i2g_file = H_FILE
V_MIN = real.horizon_row(K_FULL, PITCH) + V_MARGIN

print(f"pitch = {PITCH:+.2f} deg (negative = tilted up), horizon row {real.horizon_row(K_FULL, PITCH):.0f} px (full res)")
print(f"lane gap = {GAP_H:.2f} x height -> height {HEIGHT:.3f} m ({HEIGHT_SOURCE}), lane width {LANE_W:.2f} m")
print("saved", H_FILE)

fig, ax = plt.subplots(1, 2, figsize=(14, 3.5))
for row in costs:
    ax[0].plot(cand, row, color="gray", alpha=0.3, lw=0.8)
seen = ~np.isnan(costs).all(axis=0)                  # 두 차선이 하나도 안 잡힌 pitch 후보는 빼고
ax[0].plot(cand[seen], np.nanmean(costs[:, seen], axis=0), "b", lw=2, label="mean")
ax[0].axvline(PITCH, color="r", ls="--", label=f"best {PITCH:+.2f} deg")
ax[0].set(xlabel="pitch [deg]", ylabel="std/mean of lane gap", ylim=(0, 0.4),
          title=f"Lane parallelism over {len(static_idx)} static frames"); ax[0].legend()
ax[1].imshow(masks[0], cmap="gray"); ax[1].set_title("Yellow tape mask (above horizon ignored)"); ax[1].axis("off")
plt.tight_layout(); plt.show()
''')

md('''
### 검증: 지면 거리 격자를 영상에 투영
왼쪽은 `config.yaml` 가정 카메라(화각 90 도, 높이 0.20 m, pitch 0), 오른쪽은 실측 K + 추정 extrinsic.
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

demo_cam = real.to_camsim(undistort(real.decode_image(msgs[60])), cfg)
H_assumed = camera.build(config.load())[0]          # config.yaml 기본 카메라
H_cam = camera.build(cfg)[0]                        # h_i2g_file 이 들어간 cfg -> 실차 카메라
show(viz.side_by_side(draw_grid(demo_cam, H_assumed), draw_grid(demo_cam, H_cam)), width=1000,
     title="config.yaml assumption (hfov 90, h 0.20, pitch 0) | ost.yaml K + estimated extrinsic")
''')

md('''
### 직접 바꿔 보기 (선택)
슬라이더로 pitch 와 높이를 바꾸면 모델 입력 BEV 가 어떻게 변하는지 봄. 높이는 BEV 를 키우거나 줄이기만 하고,
pitch 는 두 차선이 벌어지거나 모이게 만듦. 바꾼 값은 저장되지 않음 (위 셀의 추정값이 계속 쓰임).
''')
code('''
try:
    from ipywidgets import interact, FloatSlider, SelectionSlider
except ImportError:
    print("ipywidgets not found: pip install ipywidgets")
else:
    cache = {i: undistort(real.decode_image(m)) for i, m in msgs.items()}

    @interact(idx=SelectionSlider(options=sorted(cache), value=60, description="frame"),
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
모델 입력은 380x300 BEV (전방 0.2~4 m, 좌우 ±1.5 m, 1 cm/px). 실차는 `real.bev_from_camera`
(보정 -> 640x400 -> `render.ipm_bev`), 시뮬은 `render.render_bev` + 카메라 가시 마스크.
같은 `H_i2g` 를 쓰니까 **회색(카메라가 못 보는 곳) 모양이 같아야 함.**

다른 건 그 안의 그림. 실차엔 바닥 반사, 조명 얼룩, 벽·의자가 바닥에 펴져서 길게 늘어진 것(평면 가정 위반)이 있음.
이게 sim-to-real 갭이고, camsim_lab 3장 증강 과제가 메우려는 것.
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
bev_real = real.bev_from_camera(undistort(real.decode_image(msgs[60])), H_I2G, cfg)
floor = np.all(bev_real == cfg.lane.color_floor, axis=-1)
print(f"BEV {bev_real.shape[1]}x{bev_real.shape[0]}, visible area: sim mask {vis_mask.mean()*100:.0f} %, "
      f"real IPM {(~floor).mean()*100:.0f} %, agreement {(vis_mask == ~floor).mean()*100:.1f} %")
show(viz.side_by_side(bev_sim, bev_real), width=700, title="sim BEV (straight section, same camera) | real BEV (frame 60)")
''')

# =========================================================================== 5. 라벨링
md('''
## 5. waypoint 자동 라벨링
정답 기준은 camsim `gt.waypoint_ahead` 와 같음: **중심선 위, 후륜축에 가장 가까운 점에서 호길이 `ahead_m` 앞의 점** (후륜축 기준 x, y).

1. 원본 해상도 영상에서 노란 테이프 마스크 -> 지면 격자로 폄 (전방 0.3~2.5 m). 먼저 펴고 색을 고르면 먼 테이프가 흐려져서 놓침
2. 조각난 마스크를 차선 둘로 묶음. 큰 조각부터 기존 차선의 y=f(x) 에 8 cm 안으로 맞으면 합침
3. 중심선: 한 차선의 각 점에서 다른 차선의 최근접점과의 중점. 같은 x 에서 평균 내면 커브에서 틀림
4. 중심선을 후륜축 쪽으로 늘려 최근접점을 찾고 (코앞은 카메라 사각지대) 호길이 `ahead_m` 지점을 waypoint 로

아래는 왼쪽부터 전면 영상(초록 = waypoint), 라벨링 격자(빨강·파랑 = 차선, 자홍 = 중심선), 모델 입력 BEV.
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
print(f"valid {frames['valid'].sum()}, jump {frames['jump'].sum()}, used for dataset {frames['use'].sum()}")

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(t_s, WPS[:, 0], label="wp_x (forward)"); ax.plot(t_s, WPS[:, 1], label="wp_y (+left)")
for n in np.flatnonzero(~frames["valid"]):
    ax.axvspan(t_s[n] - 0.013, t_s[n] + 0.013, color="r", alpha=0.3, lw=0)
ax.axvspan(0, t_s[STATIC_END], color="g", alpha=0.08)
ax.set(xlabel="time [s]", ylabel="[m]", title=f"Waypoint {cfg.waypoints.ahead_m:g} m ahead (red = invalid, green = static)")
ax.legend(); ax.grid(alpha=0.3); plt.show()
''')

# =========================================================================== 6. 저장
md('''
## 6. 데이터셋 저장 (camsim 포맷)
```
out/real_dataset/
├── images/NNNNNN.png     실차 BEV 380x300 (파일명 = bag 프레임 번호)
├── labels.csv            file, x, y, theta, wp_x, wp_y   <- DiskDataset 이 읽는 포맷 그대로
├── frames.csv            전체 프레임의 상태·플래그 (valid, jump, static)
└── real_spec.json        bag, K, D, pitch, 높이, H_i2g 등 만든 설정
```
- 실차엔 world pose 가 없어서 `labels.csv` 의 x, y, theta 는 nan. 학습엔 wp 만 쓰므로 상관없음.
- `spec.json` 은 일부러 안 씀. 그건 시뮬 데이터의 재생성 판단용이라 실차 데이터엔 해당 없음.
- 연속 프레임은 거의 같은 그림이라 `DiskDataset` 의 무작위 9:1 분할은 실차 데이터엔 안 맞음.
  평가는 `split="all"` 로, 학습에 섞을 거면 시간 구간(또는 bag) 단위로 나눌 것.
''')
code('''
if Path(DATA_OUT).exists():
    shutil.rmtree(DATA_OUT)                         # 다시 실행하면 새로 만듦
use = set(frames.loc[frames["use"], "frame"])
files = {}
with real.LabelWriter(DATA_OUT) as w:
    for i, t, msg in real.iter_images(BAG_DIR):
        if i in use:
            files[i] = w.add(real.bev_from_camera(undistort(real.decode_image(msg)), H_I2G, cfg), WPS[i], f"{i:06d}.png")
frames["file"] = frames["frame"].map(files)
frames.to_csv(f"{DATA_OUT}/frames.csv", index=False)
spec = {"bag": str(BAG_DIR), "n": len(files), "K": K_FULL.tolist(), "D": D.tolist(), "full_size": list(FULL_SIZE),
        "pitch_deg": PITCH, "height_m": HEIGHT, "height_source": HEIGHT_SOURCE, "offset_x_m": OFFSET_X_M,
        "lane_width_m": LANE_W, "ahead_m": cfg.waypoints.ahead_m, "bev": vars(cfg.bev), "H_i2g": H_I2G.tolist()}
Path(f"{DATA_OUT}/real_spec.json").write_text(json.dumps(spec, indent=1))
size_mb = sum(p.stat().st_size for p in Path(DATA_OUT).rglob("*") if p.is_file()) / 1e6
print(f"saved {len(files)} BEV images + labels.csv -> {DATA_OUT} ({size_mb:.0f} MB)")
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

pick = lab_df.sample(4, random_state=0).sort_values("file")
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
    if np.isfinite(WPS[i, 0]):
        render.draw_points(front, WPS[i], H_cam); render.draw_points_bev(bev, WPS[i], cfg)
    panel = viz.side_by_side(front, cv2.resize(bev, None, fx=400 / 380, fy=400 / 380))
    r = frames.loc[i]
    cv2.putText(panel, f"frame {i} {r.status} {'used' if r.use else 'skipped'}", (10, 25),
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
6장에서 저장한 실차 데이터를 camsim 학습 코드(`train.train`)에 그대로 넣어 봄. **파이프라인이 끝까지 도는지 보는 테스트**이고,
쓸 만한 모델을 만드는 단계는 아님. 데이터가 bag 하나 15초(400여 장)뿐이라 오래 돌리면 외워 버림.

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
GAP = 10                 # train/val 경계에서 버릴 프레임 수 (0.3 s). 경계 양쪽이 거의 같은 그림이라

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
    order = np.argsort(ds_all.files)                       # 파일명 = bag 프레임 번호 -> 시간순
    n_val = int(round(len(order) * VAL_FRAC))
    tr_idx, va_idx = order[:len(order) - n_val - GAP], order[len(order) - n_val:]
    ds_tr, ds_va = Subset(ds_all, tr_idx.tolist()), Subset(ds_all, va_idx.tolist())
    print(f"train {len(ds_tr)} (frames {ds_all.files[tr_idx[0]]}..{ds_all.files[tr_idx[-1]]}), "
          f"val {len(ds_va)} (frames {ds_all.files[va_idx[0]]}..{ds_all.files[va_idx[-1]]}), device {DEVICE}")

    t0 = time.time()
    net, hist = train.train(None, train_cfg, steps=TRAIN_STEPS, batch_size=TRAIN_BATCH, lr=TRAIN_LR, device=DEVICE,
                            out_path="out/real/model_real_test.pt", dataset=ds_tr, val_dataset=ds_va,
                            val_batches=len(ds_va) // TRAIN_BATCH + 1, log_every=max(TRAIN_STEPS // 10, 1))
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
        a.plot(frame_no[order], ds_all.wps[order, c], "k", lw=1, label="auto label")
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
    show(viz.side_by_side(*tiles), width=1000, title="val frames: green = auto label, magenta = model")
''')

# =========================================================================== 8. camsim_lab
md('''
## 8. camsim_lab 에서 쓰기
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
camsim_lab 5장이 드라이브에 올린 `model.pt` 경로를 `MODEL_PT` 에 넣으면, 실차 BEV 에서 예측한 waypoint 와 자동 라벨의 차이를 잼.
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
    show(viz.side_by_side(*tiles), width=1000, title="worst 4: green = auto label, magenta = model")
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "real_ipm_lab.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
print("wrote", os.path.normpath(out), len(cells), "cells")
