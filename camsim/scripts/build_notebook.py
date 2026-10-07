"""notebooks/camsim_lab.ipynb 생성기. 노트북 내용의 원본은 이 파일임.

노트북 고칠 일 있으면 .ipynb 말고 여기를 고치고 다시 생성할 것 (직접 고치면 다음 생성 때 덮어써짐).

    python camsim/scripts/build_notebook.py
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
# camsim 실습 — 카메라 기반 waypoint 모델

`f1tenth_gym` 은 카메라를 못 그림. 근데 바닥 테이프 트랙은 전부 평면이라, 캘리브레이션 행렬 하나로
"이 자리에서 카메라로 보면 이렇게 보인다"를 계산할 수 있음. 이 노트북은 그 지오메트리로 **BEV 학습 데이터**를
만들고, waypoint CNN 을 학습하고, gym 안에서 폐루프로 검증함.

모델이 보는 건 앞에서 본 영상이 아니라 위에서 내려다본 BEV. 시뮬은 BEV 를 바로 그리고,
실차는 카메라 영상을 IPM 으로 펴서 같은 규격의 BEV 를 만듦. 그래야 여기서 학습한 가중치가 실차로 넘어감.

**진행**: 각 장 첫 셀의 파라미터를 바꾸고 그 장을 다시 실행하면서 뭐가 달라지는지 봄.
커널 하나라 앞 장에서 만든 것(데이터, 모델)을 뒤 장에서 그대로 씀.

| 장 | 내용 | 바꿔 볼 값 |
|---|---|---|
| 1 | 카메라와 트랙 | 카메라 높이·pitch·화각, 테이프 배치, 색 |
| 2 | 정답(GT)과 데이터셋 | 정답 경로 기준(center/racing), pose 샘플링 범위, waypoint 거리, 장 수 |
| 3 | 학습 + sim-to-real 과제 | 모델 구조, 스텝, 배치, 학습률, 증강 함수 |
| 4 | 폐루프 | 속도 |
| 5 | 결과 보관 + 젯슨용 ONNX | |
''')

md('''
## 0. 설치와 설정

실행 전에 메뉴에서 두 가지. **런타임 > 런타임 유형 변경 > T4 GPU** (안 하면 학습이 10배 넘게 느림),
**파일 > 드라이브에 사본 저장** (안 하면 수정한 게 안 남음).

첫 셀이 레포를 clone 하고 의존성을 설치함 (1~2분). 다시 실행하면 pull 만 함.
`numpy.dtype size changed` 가 뜨면 런타임 재시작 후 첫 셀부터 다시.

설정은 아래 **파라미터 셀**에서 바꿈. `camsim/config.yaml` 을 직접 고치면 파라미터 셀이 덮어써서 효과 없고,
다음 실행 때 `git pull` 이 충돌함. 실수로 고쳤으면 `!git checkout -- .`
''')
code('''
%%bash
# 코랩 첫 실행이면 clone + 설치, 다시 실행이면 pull 만. 다른 fork 쓰려면 URL 만 바꿈
set -e
cd /content
[ -d f1tenth_gym ] || git clone -q --branch main https://github.com/jhcho-53/HYU_AEL_3week.git f1tenth_gym
cd f1tenth_gym
git pull -q --ff-only
pip install -q -r camsim/requirements.txt
apt-get install -y -q libgl1 > /dev/null                                     # f110_gym 이 pyglet 통해 GL 찾음
python -c "import gym, sys; sys.exit(gym.__version__ != '0.19.0')" 2>/dev/null \
  || bash camsim/scripts/install_gym019.sh    # 코랩엔 gym 0.25 가 미리 깔려 있음. f110_gym 은 0.19 API 라 갈아끼움
pip install -q --no-deps -e .                                                # numpy 는 코랩 기본값 그대로 두려고 --no-deps
''')

md('''
### 환경 확인
뭐가 깔렸고 어떤 모드가 되는지. 전부 O 여야 함. X 있으면 위 셀 출력에서 에러 찾을 것.
''')
code('''
%cd /content/f1tenth_gym
!python camsim/scripts/check_env.py
''')

md('''
### 테스트 (선택)
2~3분. 처음 한 번, 코드 pull 받은 뒤 한 번.
''')
code('''
!python -m pytest camsim/tests -q
''')

code('''
import os, sys, json, time, subprocess, numpy as np, cv2, torch, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Video, display, clear_output
os.chdir("/content/f1tenth_gym")
sys.path[:0] = ["/content/f1tenth_gym", "/content/f1tenth_gym/gym"]   # camsim, f110_gym. pip -e 의 .pth 는 재시작 전엔 안 읽힘
from camsim import config, camera, track, render, gt, augment, dataset, model, train, closed_loop, viz, handoff

plt.rcParams["axes.unicode_minus"] = False       # 코랩 기본 폰트에 한글 없음. 그래프 글자는 영어로
os.makedirs("out", exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
if DEVICE == "cpu": print("GPU 없음. 런타임 > 런타임 유형 변경 > T4 GPU 로 바꾸고 처음부터 다시 돌릴 것")

def show(img_bgr, width=640, title=None):
    if title: print(title)
    display(Image(data=cv2.imencode(".png", img_bgr)[1].tobytes(), width=width))
''')

md('''
### 파라미터
`camsim/config.yaml` 이 기본값이고 이 셀이 그 위에 덮어씀. **값을 바꿨으면 이 셀부터 다시 실행** (여기서 `cfg` 를 새로 만듦).
어디까지 다시 돌릴지는 뭘 바꿨느냐에 달림:

| 바꾼 것 | 다시 실행 |
|---|---|
| `model.arch`, `STEPS`, `BATCH`, `LR`, `AUGMENT_FN` | 이 셀 → 3장 |
| `speed_mps` | 이 셀 → 4장 |
| 카메라, 테이프 배치, `waypoints.line`, `ahead_m`, `sampling.*` | 이 셀 → 2장(설정 바뀐 걸 알아채고 데이터 다시 만듦) → 3장 → 4장 |
''')
code('''
cfg = config.load()

# ---- 1장: 카메라와 트랙 ----
cfg.camera.height_m   = 0.20      # 바닥에서 렌즈 중심까지 (m). 마운트 확정 전 가정값
cfg.camera.pitch_deg  = 0.0       # 아래로 숙인 각도 (+ = 아래). 정면이면 0
cfg.camera.hfov_deg   = 90.0      # 렌즈 수평 화각
cfg.lane.follow_walls  = True     # True 면 맵 벽 안쪽을 따라 테이프 놓음 (폭이 구간마다 다름)
cfg.lane.wall_margin_m = 0.25     # 벽에서 얼마나 안쪽에 놓을지
cfg.lane.track_width_m = 0.8      # follow_walls = False 일 때 좌우 테이프 간격
cfg.lane.color_floor  = [128, 128, 128]   # BGR. 회색 바닥
cfg.lane.color_tape   = [0, 220, 255]     # BGR. 노란 테이프

# ---- 2장: GT 와 데이터 ----
cfg.waypoints.line = "center"     # 정답 경로 기준. "center" = 좌우 테이프 중간선 (실차 라벨링과 같음)
                                  #                "racing" = CSV 의 레이싱 라인 (코너 안쪽)
cfg.waypoints.ahead_m = 1.0       # 예측할 waypoint 의 전방 거리 (m). 이 점이 곧 pure pursuit 목표점. 2 m 넘기지 말 것
cfg.sampling.lateral_frac = 0.35  # 기준선에서 ±트랙폭*frac 범위에 pose 뿌림
cfg.sampling.heading_deg  = 15.0  # 헤딩도 ± 이만큼
N_DATASET = 10000                 # 저장할 장 수. resnet18 은 파라미터가 1,200만이라 1000 장이면 외워 버림
DATA_DIR = "out/dataset"

# ---- 3장: 학습 ----
cfg.model.arch = "resnet18"       # "resnet18" (ImageNet 사전학습) | "small" (작은 CNN, 젯슨이 느리면)
cfg.model.pretrained = True
STEPS, BATCH, LR = 3000, 32, 3e-4  # 사전학습 가중치라 LR 을 낮게. 1e-3 이면 초반에 망가짐

# 증강 세기. 기본 학습은 증강 없이(plain) 하고, 아래 값은 3장 sim-to-real 과제에서 씀
cfg.augment.pitch_jitter_deg = 2.0    # 가감속 때 pitch 변해서 BEV 휘는 정도 (도)
cfg.augment.ipm_blur_max_px = 5       # 먼 곳일수록 커지는 블러 (px. BEV 1 px = 1 cm)
cfg.augment.tape_dropout_prob = 0.3   # 테이프 마모·가림
cfg.augment.brightness_delta = 40     # 노출 변화
cfg.augment.contrast_range = [0.8, 1.2]
cfg.augment.gamma_range = [0.7, 1.5]
cfg.augment.hue_shift_deg = 15        # 조명 색온도
cfg.augment.sat_scale = [0.7, 1.3]
cfg.augment.illum_strength = 0.3      # 불균일 조명
cfg.augment.shadow_prob = 0.3
cfg.augment.blur_max_px = 3
cfg.augment.noise_sigma = 6
cfg.augment.jpeg_quality = [40, 90]   # image_transport compressed 아티팩트

# ---- 4장: 폐루프 ----
cfg.closed_loop.speed_mps = 2.0
cfg.closed_loop.max_steps = 4000      # 제어 틱. 25 Hz 라 160 초에서 끊음

trk = track.from_csv(cfg.closed_loop.centerline_csv, cfg)
H_g2i, H_i2g = camera.build(cfg)
print(f"트랙 길이 {trk.length:.1f} m, 테이프 사각형 {len(trk.quads)}개, 초점거리 {camera.focal_px(cfg):.0f} px")
''')

# =========================================================================== 1. 카메라와 트랙
md('''
## 1. 카메라와 트랙
지면(z=0)과 이미지 사이는 homography `H` 하나로 닫힘. 실차 IPM 에서 구하는 `H_i2g` 의 역행렬이 곧 합성 카메라.
높이 0.2 m 에 정면(pitch 0)으로 달면 지평선이 이미지 세로 한가운데 옴. `pitch_deg` 와 `height_m` 을 바꿔
다시 실행하면 지평선이 어디로 가는지, 가까운 바닥이 어디부터 보이는지 확인 가능.
''')
code('''
horizon_v = camera.project(H_g2i, np.array([[1000.0, 0.0]]))[0, 1]
near_x = camera.project(H_i2g, np.array([[cfg.camera.image_width / 2, cfg.camera.image_height]]))[0, 0]
print(f"지평선 행 v = {horizon_v:.0f} px (이미지 높이 {cfg.camera.image_height}), 가장 가까운 보이는 바닥 = {near_x:.2f} m")

i = 50
pose = np.array([*trk.center[i], trk.heading[i]])
wp = gt.waypoint_ahead(pose, trk, cfg)
img_raw = render.render(pose, trk.quads, None, H_g2i, cfg)      # 순수 카메라 뷰. 아래 IPM 비교에 씀
show(render.draw_points(img_raw.copy(), wp, H_g2i), title="합성 카메라 뷰 (초록 = GT waypoint). 모델은 이걸 안 보고, 이걸 펴서 만든 BEV 를 봄")
''')

md('''
### 트랙 어디인가
차 주변을 위에서 본 그림. 검은 띠 = gym 맵의 벽, 하늘색 부채꼴 = 카메라 화각, 보라 사각형 = BEV 범위, 초록 = GT waypoint.

실차 트랙엔 벽이 없고 테이프가 경계. 벽은 gym 충돌과 LiDAR 에만 쓰이고 카메라 모델은 테이프만 봄.
`follow_walls = True` 면 테이프를 맵 벽 안쪽 `wall_margin_m` 지점에 놓아서 트랙 모양이 맵과 같아짐 (대신 폭이 구간마다 다름).
`False` 면 중심선에서 `track_width_m/2` 로 일정하게 놓음 (실습실 테이프 트랙 방식).
''')
code('''
mapimg = viz.MapImage(cfg.closed_loop.map_yaml)
show(viz.local_view(pose, trk, wp, cfg, mapimg), width=500, title="차 주변 top-down (위 = 차량 전방)")
overview, off = viz.draw_track_on_map(mapimg, trk, cfg)
viz.mark_poses_on_map(overview, off, mapimg, [pose])
show(overview, width=600, title="전체 맵에서의 위치")
''')

md('''
### BEV 세 장: 정답 / 시뮬 모델 입력 / 실차 IPM
왼쪽은 지오메트리에서 바로 그린 top-down 이라 참값.
가운데는 거기에 **카메라 가시 마스크**를 씌운 것. 카메라가 못 보는 코앞과 화각 밖을 바닥색으로 덮음. 이게 **시뮬 학습 데이터**.
오른쪽은 위 카메라 뷰를 `H_i2g` 로 편 것, 즉 실차 IPM 이 만들어 낼 BEV.

**가운데와 오른쪽이 같은 모양이어야** 시뮬에서 학습한 가중치가 실차로 넘어감.
멀어질수록 오른쪽이 늘어지고 끊기는데 그게 IPM 해상도 한계. 이 카메라(높이 0.2 m)로는 1.5 m 까지 테이프 위치가
0.5 cm 안에서 복구되고, 2 m 넘으면 코너에서 7 cm 씩 튐. waypoint 를 1 m 에 두는 이유.
''')
code('''
vis_mask = render.bev_visibility_mask(H_g2i, cfg)
bev_true = render.render_bev(pose, trk.quads, cfg)
bev_sim = render.render_bev(pose, trk.quads, cfg, vis_mask)      # 모델 입력
bev_ipm = render.ipm_bev(img_raw, H_i2g, cfg)
show(viz.side_by_side(bev_true, bev_sim, bev_ipm), width=900, title=f"정답 | 시뮬 모델 입력 | 실차 IPM   (전방 {cfg.bev.x_range_m} m, 좌우 {cfg.bev.y_range_m} m, {cfg.bev.resolution_m*1000:.0f} mm/px)")
print(f"BEV 크기 {bev_sim.shape[1]} x {bev_sim.shape[0]} px, 카메라가 보는 비율 {vis_mask.mean()*100:.0f} %")

# 왕복 오차. 카메라 뷰의 테이프 픽셀을 지면으로 되돌리면 원래 테이프에서 얼마나 벗어나나
vs, us = np.where(np.all(img_raw == cfg.lane.color_tape, axis=-1))
g = camera.project(H_i2g, np.column_stack([us, vs]).astype(float))
qv = render.to_vehicle(pose, trk.quads).reshape(-1, 2)
d = np.sqrt(((g[:, None, :] - qv[None, ::4, :]) ** 2).sum(-1)).min(1)
print(f"IPM 왕복 오차: 중앙값 {np.median(d)*100:.1f} cm, 95% {np.percentile(d, 95)*100:.1f} cm")
''')

# =========================================================================== 2. GT와 데이터
md('''
## 2. 정답(GT)과 데이터셋
**주행하면서 데이터를 모으지 않음.** 주행 가능 영역에 pose 를 무작위로 뿌리고, 각 자리에서 BEV 를 그리고,
기준선을 따라 전방 호길이로 waypoint 를 뽑음. 그래서 "차선 이탈 직전" 같은 상황도 데이터에 자연히 들어가고,
모델은 복귀 동작을 따로 라벨링하지 않아도 배움.

**정답 경로 기준** (`cfg.waypoints.line`)
- `center` — 좌우 테이프의 중간선. 실차에서 HSV+IPM 으로 자동 라벨링할 때와 같은 기준이라 그대로 넘어감.
- `racing` — CSV 의 레이싱 라인. 코너 안쪽을 파고들어 랩타임이 짧지만, 실차 라벨을 같은 기준으로 만들려면
  트랙마다 따로 최적화해야 함.

두 기준으로 각각 학습해서 4장에서 랩타임과 완주 안정성을 비교해 볼 것. 기준을 바꾸면 2장이 데이터를 다시 만듦.
아래는 왼쪽부터 참고용 카메라 뷰 / 모델이 보는 BEV / 차 주변 top-down.
''')
code('''
rng = np.random.default_rng(0)
poses = []
for n in range(4):
    bev, wp_s, p, cam = dataset.make_sample(trk, cfg, rng, with_camera=True, mask=vis_mask)
    poses.append(p)
    render.draw_points(cam, wp_s, H_g2i); render.draw_points_bev(bev, wp_s, cfg)
    show(viz.side_by_side(cam, bev, viz.local_view(p, trk, wp_s, cfg, mapimg)), width=1000, title=f"샘플 {n}: 카메라 뷰(참고) | BEV 모델 입력 | 위치")
overview, off = viz.draw_track_on_map(mapimg, trk, cfg)
show(viz.mark_poses_on_map(overview, off, mapimg, poses), width=600, title="샘플들의 위치")
''')

md('''
### 데이터셋 저장
`out/dataset/images/NNNNNN.png` (BEV) 와 `labels.csv` (pose, waypoint) 로 저장. 증강은 저장 안 하고 학습 로딩 때
함수 하나(`augment_fn`)로 넣음. 같은 데이터로 증강만 바꿔 가며 비교하려고.
BEV 는 380×300 (1 px = 1 cm). 10,000장이면 1~2분, 디스크 90 MB 정도. 드라이브 말고 **코랩 로컬 디스크**에 만들 것.
같이 저장되는 `spec.json` 덕에 설정이 바뀌면 다음 실행 때 알아서 다시 만듦.

실차에서도 IPM 으로 만든 BEV 를 같은 포맷으로 저장하면 아래 코드가 그대로 돎.
''')
code('''
if dataset.needs_regeneration(DATA_DIR, cfg, N_DATASET):      # 없거나, 다른 설정·장 수로 만든 데이터면
    t0 = time.time()
    dataset.generate_dataset(trk, cfg, N_DATASET, DATA_DIR, seed=0, log_every=5000)
    print(f"{N_DATASET}장 생성, {time.time() - t0:.0f}s")
else:
    print(f"{DATA_DIR} 재사용 (같은 설정으로 만든 데이터)")
files, ds_poses, ds_wps = dataset.read_labels(DATA_DIR)
print(f"데이터 {len(files)}장")
display(pd.read_csv(f"{DATA_DIR}/labels.csv").head())
''')

md('''
### 데이터 훑어보기
저장된 BEV 에 라벨(초록)을 찍어 봄. 라벨이 테이프 사이 한가운데를 따라가는지, 횡 오프셋과 헤딩 분포가
의도한 범위인지 확인. 여기서 이상하면 학습 아무리 돌려도 소용없음.
''')
code('''
for i in np.random.default_rng(1).choice(len(files), 3, replace=False):
    im = cv2.imread(f"{DATA_DIR}/images/{files[i]}")
    show(render.draw_points_bev(im, ds_wps[i], cfg), width=360, title=f"{files[i]}  pose={np.round(ds_poses[i], 2)}")
lat = np.array([gt.lateral_error(trk, p[:2]) for p in ds_poses])
dth = np.rad2deg(np.angle(np.exp(1j * (ds_poses[:, 2] - trk.heading[[gt.nearest_index(trk, p[:2]) for p in ds_poses]]))))
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].hist(lat, 30); ax[0].set_xlabel("lateral offset from centerline (m)")
ax[1].hist(dth, 30); ax[1].set_xlabel("heading error (deg)")
plt.tight_layout(); plt.show()
''')

# =========================================================================== 3. 학습
md('''
## 3. 학습
BEV 를 받아 waypoint 하나의 (x, y) 좌표(m)를 냄. 손실은 Huber.

```
BEV (3, 380, 300) BGR  →  ResNet-18 백본 (ImageNet 사전학습)  →  (512, 12, 10)
                       →  1×1 conv 512→32  →  flatten 3840  →  FC 256  →  FC 2 = (x, y)
```

- **사전학습 백본**: 시뮬 BEV 는 깨끗한 노란 줄뿐이라, 처음부터 배운 모델은 실차 조명·질감 변화에 무너지기 쉬움.
  ImageNet 특징이 그런 변화에 더 강함. 이득은 시뮬 오차보다 3장 열화 표에서 보임.
- **pooling 없는 head**: 점이 **어디** 있는지가 답이라 위치 정보를 평균 내 버리면 안 됨. 그리고 젯슨 TensorRT 용 ONNX 로
  깨끗하게 넘어가야 해서 AdaptiveAvgPool 을 안 씀 (입력 크기에 따라 변환이 실패함).
- **BGR 입력**: ImageNet 가중치는 RGB 로 학습됐는데, conv1 의 입력 채널 순서를 뒤집어서 BGR 을 그대로 받게 함.

`cfg.model.arch = "small"` 이면 작은 CNN (37만 파라미터). 학습이 훨씬 빠르고, 젯슨이 느리면 이걸로.

train/val 은 9:1. **train loss 만 내려가고 val loss 가 멈추면 오버피팅.** 스텝과 장 수를 바꿔 비교해 볼 것.
연한 선이 원 값, 진한 선이 5점 이동평균. val 은 매번 약 1,000장으로 재는데도 좀 흔들림 — 샘플마다 난이도가 달라서
(예: pitch 지터로 먼 테이프가 잘린 것).
''')
code('''
AUGMENT_FN = None        # plain 학습. 과제에서 my_augment 로 바꿔 재학습
ds_train = dataset.DiskDataset(DATA_DIR, cfg, "train", val_frac=0.1, augment_fn=AUGMENT_FN)
ds_val = dataset.DiskDataset(DATA_DIR, cfg, "val", val_frac=0.1)
print(f"train {len(ds_train)}장, val {len(ds_val)}장, device {DEVICE}")

def smooth(y, k=5):      # 이동평균. 양끝은 있는 만큼만
    return np.array([y[max(0, i - k + 1):i + 1].mean() for i in range(len(y))])

def live_plot(history):
    clear_output(wait=True)
    st = [h["step"] for h in history]
    plt.figure(figsize=(7, 3.5))
    tr = np.array([h["loss"] for h in history])
    plt.plot(st, tr, color="C0", alpha=.35); plt.plot(st, smooth(tr), color="C0", label="train (smoothed)")
    if "val_loss" in history[-1]:
        va = np.array([h["val_loss"] for h in history])
        plt.plot(st, va, color="C1", alpha=.35); plt.plot(st, smooth(va), color="C1", label="val (smoothed)")
    plt.yscale("log"); plt.xlabel("step"); plt.ylabel("Huber loss"); plt.grid(alpha=.3); plt.legend()
    plt.title(f"step {st[-1]}  train {history[-1]['loss']:.4f}" + (f"  val {history[-1]['val_loss']:.4f}" if "val_loss" in history[-1] else ""))
    plt.show()

net, hist = train.train(trk, cfg, steps=STEPS, batch_size=BATCH, lr=LR, device=DEVICE, out_path="model.pt",
                        num_workers=2 if DEVICE == "cuda" else 0, log_every=max(STEPS // 30, 1),
                        dataset=ds_train, val_dataset=ds_val, val_batches=32, callback=live_plot)
print("model.pt 저장")
''')

md('''
### 오차
예측점과 정답점 사이 거리(cm). 테이프 폭이 5 cm 니까 대부분 그 안에 들어오면 충분함.
히스토그램 둘(저장된 val 이미지 / 새로 뽑은 pose)이 겹쳐야 정상.
''')
code('''
pred = model.Predictor(net, cfg, DEVICE)
r_val = train.evaluate_dataset(pred, ds_val, n=300)
r_new = train.evaluate(pred, trk, cfg, n=300)
print(f"val (저장 이미지 {r_val['n']}장): 평균 {r_val['mean_m']*100:.1f} cm, 최대 {r_val['max_m']*100:.1f} cm")
print(f"새 pose {r_new['n']}개:           평균 {r_new['mean_m']*100:.1f} cm, 최대 {r_new['max_m']*100:.1f} cm")
plt.figure(figsize=(7, 3.5))
plt.hist(r_val["errs_m"] * 100, 30, alpha=.6, label="val, saved images")
plt.hist(r_new["errs_m"] * 100, 30, alpha=.6, label="fresh poses")
plt.axvline(cfg.lane.tape_width_m * 100, color="r", ls="--", label="tape width")
plt.xlabel("waypoint error (cm)"); plt.ylabel("count"); plt.legend(); plt.grid(alpha=.3); plt.show()

bev, wp_s, p, cam = dataset.make_sample(trk, cfg, np.random.default_rng(7), with_camera=True, mask=vis_mask)
wp_pred = pred.predict(bev)
render.draw_points_bev(bev, wp_s, cfg, (0, 255, 0)); render.draw_points_bev(bev, wp_pred, cfg, (255, 0, 255))
render.draw_points(cam, wp_s, H_g2i, (0, 255, 0)); render.draw_points(cam, wp_pred, H_g2i, (255, 0, 255))
show(viz.side_by_side(cam, bev), width=900, title="초록 = 정답, 자홍 = 모델 예측  (왼쪽은 참고용, 모델은 오른쪽 BEV 만 봄)")
''')

md('''
### sim-to-real 갭: plain 모델은 실차에서 무너짐
위 모델은 깨끗한 시뮬 BEV 에서만 잘 맞음. 실차 IPM 출력엔 시뮬에 없는 것들이 섞임.

| 실차에서 생기는 일 | BEV 에 나타나는 모양 | 함수 |
|---|---|---|
| 가감속 때 차체 pitch 변화 (서스펜션) | IPM 평면 가정이 깨져 먼 곳이 휨 | `jitter_bev` (기하학적으로 정확) |
| IPM 원거리 해상도 저하 | 먼 테이프가 뭉개짐 (1장 오른쪽 그림) | `ipm_blur` |
| 테이프 마모·벗겨짐, 다른 차의 가림 | 테이프 일부가 사라짐 | `erase_patches` |
| 조도 변화, 노출 변동 | 전체가 밝거나 어두워짐 | `brightness_contrast`, `gamma` |
| 조명 색온도 (형광등/자연광) | 테이프 색조가 달라짐 | `hsv_shift` |
| 천장 조명이 한쪽만 밝음 | 화면을 가로지르는 밝기 기울기 | `illumination` |
| 구조물·사람 그림자 | 일부 영역이 어두워짐 | `shadow` |
| 게인 올린 센서, 초점 흐림 | 노이즈, 블러 | `noise`, `blur` |
| `image_transport` compressed | JPEG 블록 아티팩트 | `jpeg` |
| 캘리브레이션 오차 | BEV 전체가 살짝 기울거나 늘어남 | (직접 작성) |

실차 데이터 없어도 확인 가능. 같은 모델을 **열화된 BEV** 로 평가해서 오차가 얼마나 튀는지 볼 것.
`model.arch` 를 `small` 로 바꿔 다시 학습하고 이 표를 비교하면, 사전학습 백본이 실제로 버티는지 보임.
''')
code('''
degradations = {
    "clean": None,
    "pitch jitter": lambda b, r: augment.jitter_bev(b, cfg, r),
    "ipm blur (far)": lambda b, r: augment.ipm_blur(b, cfg, r),
    "tape erased": lambda b, r: augment.erase_patches(b, cfg, r),
    "lighting": lambda b, r: augment.gamma(augment.brightness_contrast(augment.illumination(b, cfg, r), cfg, r), cfg, r),
    "shadow": lambda b, r: augment.shadow(b, cfg, r),
    "sensor (blur+noise+jpeg)": lambda b, r: augment.jpeg(augment.noise(augment.blur(b, cfg, r), cfg, r), cfg, r),
    "all (example_augment)": lambda b, r: augment.example_augment(b, cfg, r),
}
rows = {k: train.evaluate(pred, trk, cfg, n=200, degrade_fn=f) for k, f in degradations.items()}
tbl = pd.DataFrame({"mean (cm)": {k: r["mean_m"] * 100 for k, r in rows.items()},
                    "max (cm)": {k: r["max_m"] * 100 for k, r in rows.items()}}).round(1)
print("열화 종류별 waypoint 오차. clean 대비 얼마나 나빠지나:")
display(tbl)

# 각 열화가 BEV 를 실제로 어떻게 바꾸는지도 눈으로
base = dataset.make_sample(trk, cfg, np.random.default_rng(3), mask=vis_mask)[0]
tiles = [base] + [f(base, np.random.default_rng(2)) for k, f in degradations.items() if f]
names = ["clean"] + [k for k, f in degradations.items() if f]
for i in range(0, len(tiles), 4):
    show(viz.side_by_side(*tiles[i:i + 4]), width=1000, title=" | ".join(names[i:i + 4]))
''')

md('''
### 과제: sim-to-real 증강 설계
`my_augment(bev, rng)` 를 써서 위 표의 열화들(그리고 표에 없는 실차 현상)에 강한 모델을 만들 것.

1. 입력과 출력은 같은 크기의 BEV (BGR uint8). 라벨은 안 바뀌므로 **정답을 옮기는 변형(이동·회전)은 금지.**
   정답을 유지하는 관측 변화만 허용.
2. 아래 셀의 `AUGMENT_FN = my_augment` 로 두고 3장 첫 셀부터 다시 실행해 재학습.
3. 위 열화 표를 다시 뽑아 plain 모델과 비교하고, 4장 폐루프에서 완주 여부와 횡오차 비교.
4. 어떤 열화가 폐루프 실패로 이어지는지, 어떤 증강이 그걸 막았는지 한 문단으로 정리.

`camsim/augment.py` 함수를 조합해도 되고 OpenCV 로 직접 써도 됨. 증강 과하게 넣으면 clean 성능이 깎이니 표로 확인할 것.
`jitter_bev` 빼고 나머지는 실제 카메라 물리를 대충 근사한 것. 실차 영상 찍고 나면 실제로 뭐가 깨지는지 보고 다시 설계하는 게 맞음.
''')
code('''
def my_augment(bev, rng):
    # 여기를 채움. 주석 풀어 써도 되고 직접 OpenCV 로 짜도 됨. 세기는 파라미터 셀의 cfg.augment.*
    out = bev
    # --- 기하 (라벨은 그대로 두고 관측만 바꿈) ---
    # out = augment.jitter_bev(out, cfg, rng)          # pitch 변화 -> BEV 휨
    # out = augment.ipm_blur(out, cfg, rng)            # 먼 곳일수록 뭉개짐
    # out = augment.erase_patches(out, cfg, rng)       # 테이프 마모·가림
    # --- 조명 ---
    # out = augment.illumination(out, cfg, rng)        # 불균일 조명
    # out = augment.shadow(out, cfg, rng)              # 그림자
    # out = augment.brightness_contrast(out, cfg, rng) # 노출
    # out = augment.gamma(out, cfg, rng)               # 감마
    # out = augment.hsv_shift(out, cfg, rng)           # 색온도
    # --- 센서 ---
    # out = augment.blur(out, cfg, rng)
    # out = augment.noise(out, cfg, rng)
    # out = augment.jpeg(out, cfg, rng)
    # --- 직접 만들어 보기 ---
    # out = cv2.GaussianBlur(out, (5, 5), 0)
    # out = cv2.LUT(out, np.clip((np.arange(256) / 255) ** 0.8 * 255, 0, 255).astype(np.uint8))
    return out

AUGMENT_FN = my_augment      # 이렇게 두고 3장 첫 셀(ds_train)부터 다시 실행하면 증강 학습
sample_bev = dataset.make_sample(trk, cfg, np.random.default_rng(3), mask=vis_mask)[0]
show(viz.side_by_side(sample_bev, my_augment(sample_bev, np.random.default_rng(0))), width=600, title="my_augment 미리보기: 원본 | 증강")
''')

# =========================================================================== 4. 폐루프
md('''
## 4. 폐루프 검증 (gym, ROS 없음)
제어 틱마다 BEV 렌더 → 모델 추론 → pure pursuit → `env.step`.
종료 조건은 실차 규칙과 같음. **차체 모서리가 테이프를 넘으면 실격**(`tape_crossed`), 한 바퀴 돌면 `lap`.

횡오차 그래프의 빨간 점선이 테이프 한계선. 곡선이 여기 닿으면 실격. 트랙 폭이 구간마다 다르면 선도 같이 움직임.
영상은 왼쪽 카메라 뷰(참고), 오른쪽 모델 입력 BEV, 초록 = 모델이 낸 waypoint.
''')
code('''
env = closed_loop.make_env(cfg)
r = closed_loop.run(env, pred, trk, cfg, H_g2i, video_path="out/run.mp4")
print(f"{r.reason}  진행 {r.progress_m:.1f} / {trk.length:.1f} m  시간 {r.time_s:.1f} s  "
      f"횡오차 평균 {r.mean_lateral_m*100:.1f} cm  최대 {r.max_lateral_m*100:.1f} cm")

t = np.arange(r.steps) / r.control_hz_eff
idx = [gt.nearest_index(trk, p[:2]) for p in r.pose_trace]
limit = np.minimum(trk.left_m[idx], trk.right_m[idx]) - cfg.lane.tape_width_m / 2 - cfg.closed_loop.car_width_m / 2
plt.figure(figsize=(8, 3.5))
plt.plot(t, r.lateral_trace * 100, label=f"model -> {r.reason}")
plt.plot(t, limit * 100, "r--", label="body touches tape")
plt.xlabel("time (s)"); plt.ylabel("lateral error (cm)"); plt.legend(fontsize=8); plt.grid(alpha=.3); plt.show()
display(Video(viz.to_h264("out/run.mp4"), embed=True, width=900))   # mp4v 는 브라우저가 못 읽어서 H.264 로 변환
''')

md('''
### 맵 위의 주행 경로
위 그래프는 얼마나 벗어났는지만 보여주고 **트랙 어디서**인지는 안 보여줌. 그래서 경로를 맵에 그림.
회색 실선 = 오라클(정답 waypoint 그대로 달린 것), 색 파선 = 모델. 모델만 벗어나는 코너면 모델 탓,
오라클도 같이 벗어나면 그 코너는 이 속도·목표점으로 원래 무리.

163 m 트랙에서 차이는 수 cm 라 실제 축척으론 겹쳐 보여서 **중심선 대비 편차만 `MAGNIFY` 배로 과장**함 (실제 위치 아님).
실격했으면 그 지점을 실제 축척으로 확대해서 한 장 더 보여줌.
''')
code('''
MAGNIFY = 15      # 편차를 몇 배로 과장할지. 1 = 실제 축척

r_or = closed_loop.run(env, model.OraclePredictor(trk, cfg), trk, cfg, H_g2i)
paths = {f"oracle -> {r_or.reason} ({r_or.progress_m:.0f} m)": r_or.pose_trace,
         f"model -> {r.reason} ({r.progress_m:.0f} m, {r.time_s:.0f} s)": r.pose_trace}
pm, _ = viz.draw_paths_on_map(mapimg, trk, cfg, paths, magnify=MAGNIFY)
show(pm, width=900, title="전체 맵: 오라클 vs 모델")

if r.reason != "lap":
    plain, off = viz.draw_paths_on_map(mapimg, trk, cfg, paths, legend=False)
    show(viz.crop_around(plain, off, mapimg, r.pose_trace[-1], half_m=3.0, scale=3), width=500,
         title=f"모델이 멈춘 지점 ({r.reason}) — 실제 축척")
''')

# =========================================================================== 5. 보관
md('''
## 5. 결과 보관 + 젯슨용 ONNX
여기까지 만든 파일은 코랩 VM 에 있고 런타임 끊기면 사라짐. 아래 셀이 드라이브로 옮김.

젯슨은 **ONNX** 를 받아서 TensorRT 엔진을 직접 만듦. 엔진은 GPU·TRT 버전마다 달라서 코랩에서 만든 건 못 씀.
ONNX 안에 학습된 가중치가 다 있어서 **젯슨엔 PyTorch 가 필요 없음.**

1. 아래 셀이 `model.onnx` 를 만들고, onnxruntime 으로 돌려서 PyTorch 결과와 같은지 확인한 뒤 드라이브에 올림.
   `checkpoint.json` 에 SHA-256 과 설정·git commit 을 같이 기록함. `model.pt` 도 재학습용으로 같이 올림.
2. 드라이브 웹에서 **`model.onnx`** 공유를 "링크가 있는 모든 사용자: 뷰어"로 바꾸고 링크 복사.
3. 젯슨에서 (`LINK`, `SHA` 는 바꿀 것):
   ```bash
   pip install gdown && gdown 'LINK' -O model.onnx
   echo 'SHA  model.onnx' | sha256sum -c -
   /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
   ```
   엔진 입력은 `bev` (1, 3, 380, 300) BGR 0~1 float, 출력은 `wp` (1, 2) = (x, y) m.
''')
code('''
from pathlib import Path
import shutil, onnxruntime as ort

SAVE_DATASET = False    # 데이터셋 zip 까지 보관하려면 True

onnx_path = model.export_onnx(net, cfg, "model.onnx")
x = torch.rand(1, 3, *render.bev_size(cfg))
with torch.no_grad():
    ref = net.cpu().eval()(x).numpy()
out = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(None, {"bev": x.numpy()})[0]
print(f"ONNX vs PyTorch 최대 차이: {np.abs(out - ref).max():.1e}")     # 1e-4 보다 작아야 정상
net.to(DEVICE)

from google.colab import drive
drive.mount("/content/drive")
dst = Path("/content/drive/MyDrive/camsim_results")
git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
manifest = handoff.export_checkpoint(Path(onnx_path), dst, cfg, git_commit)
shutil.copy2("model.pt", dst / "model.pt")
if SAVE_DATASET:
    archive = shutil.make_archive("out/dataset", "zip", root_dir="out", base_dir="dataset")
    shutil.copy2(archive, dst / "dataset.zip")
print("드라이브:", dst / "model.onnx")
print("SHA-256:", manifest["sha256"])
print("git commit:", git_commit)
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "camsim_lab.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
print("wrote", os.path.normpath(out), len(cells), "cells")
