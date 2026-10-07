"""notebooks/week3_label_train.ipynb 생성기. 노트북 내용의 원본은 이 파일임.

노트북 고칠 일 있으면 .ipynb 말고 여기를 고치고 다시 생성할 것 (직접 고치면 다음 생성 때 덮어써짐).

    python camsim/scripts/build_week3_notebook.py
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
# 3주차 실습 — rosbag → 클릭 라벨링 → 학습

조교 차가 실습실 트랙을 달리며 녹화한 **rosbag** 에서 BEV 를 만들고, 프레임마다 정답(1 m 앞 점)을 **직접 찍어서**
`camsim_lab` 과 같은 형식의 데이터셋을 만든 뒤 같은 학습 코드를 돌려 봄.

시뮬(`camsim_lab`)은 정답이 자동으로 나왔음. 실차는 아무도 정답을 모름. `real_ipm_lab` 5장은 노란 테이프를 찾아
정답을 **자동으로** 만들었는데, 테이프가 안 보이거나 하나만 보이면 틀리거나 포기함. 그래서 사람이 찍는 단계가 필요함.
오늘 할 일은 그 과정 전체이고, 4장에서 내가 찍은 점과 자동 라벨을 비교해 봄.

| 장 | 내용 |
|---|---|
| 0 | 설치, 파라미터 |
| 1 | rosbag 살펴보기 |
| 2 | BEV 만들기 (`real_ipm_lab` 과 같은 IPM) |
| 3 | 라벨링 (클릭) |
| 4 | 자동 라벨과 비교 |
| 5 | 데이터셋 (camsim 형식) |
| 6 | 학습 |

라벨이 30장 정도라 성능은 보지 않음. **녹화 → 라벨 → 데이터셋 → 학습이 끝까지 돌아가는 것**이 목표.
''')
md('''
## 0. 설치

실행 전에 메뉴에서 두 가지. **런타임 > 런타임 유형 변경 > T4 GPU**, **파일 > 드라이브에 사본 저장**.

첫 셀이 레포를 clone 함. 오늘 쓸 bag(`examples/week3_bag/`, 약 60 MB)이 레포에 같이 들어 있어서 따로 받을 것 없음.
''')
code('''
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/jhcho-53/HYU_AEL_3week.git"     # 다른 fork 쓰려면 여기만 바꿈
if IN_COLAB:
    if not os.path.isdir("/content/f1tenth_gym"):
        subprocess.run(["git", "clone", "-q", "--branch", "main", REPO_URL, "/content/f1tenth_gym"], check=True)
    subprocess.run(["git", "-C", "/content/f1tenth_gym", "pull", "-q", "--ff-only"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rosbags==0.11.5"], check=True)   # ROS 없이 bag 읽기
print("colab" if IN_COLAB else "local")
''')
code('''
import numpy as np, cv2, torch, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/content/f1tenth_gym") if IN_COLAB else next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "camsim").is_dir())
os.chdir(ROOT)
sys.path[:0] = [str(ROOT)]
from camsim import config, dataset, model, train, render, week3_lab as lab

plt.rcParams["axes.unicode_minus"] = False       # 코랩 기본 폰트에 한글 없음. 그래프 글자는 영어로
os.makedirs("out", exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = config.load()                              # camsim 기본 설정 (BEV 300 x 380 px, 1 px = 1 cm, 1 m 앞 waypoint)
print("device:", DEVICE)
''')
md('''
### 파라미터
값을 바꿨으면 이 셀부터 다시 실행. 내 bag 으로 할 때 바꾸는 곳 (맨 아래 "내 bag 으로 하기").

- `BAG`, `OST`: bag 과 그 카메라의 렌즈 캘리브레이션(`ost.yaml`, 1주차 결과)
- `PITCH_DEG`, `HEIGHT_M`: 카메라가 바닥에서 얼마나 높이, 어떤 각도로 달렸는지. `real_ipm_lab` 3장이 이 bag 의 원본에서
  추정한 값을 그대로 적어 둠 (차선 간격 0.80 m 실측 기준). 그 노트북을 먼저 돌릴 필요는 없음
- `LANE_WIDTH_M`: 테이프 중심 사이 간격. 4장의 자동 라벨에만 씀
''')
code('''
BAG = lab.SAMPLE / "run_train2_part1_2hz"        # 폴더, 또는 .db3 파일 하나
OST = lab.SAMPLE / "ost.yaml"                    # 그 카메라의 K, 왜곡 계수
TOPIC = lab.IMAGE_TOPIC                          # 카메라 영상 토픽 (/flir_camera/image_raw)
PITCH_DEG = -4.25                                # real_ipm_lab 3장 결과 (음수 = 살짝 위로 들림)
HEIGHT_M = 0.178                                 # real_ipm_lab 3장 결과 (m)
OFFSET_X_M = 0.0                                 # 후륜축 -> 카메라 전방 거리. 모르면 0
LANE_WIDTH_M = 0.8                               # 테이프 중심 간 간격 (m, 실측)
NAME = Path(BAG).stem                            # 라벨 파일과 데이터셋 이미지 이름에 씀
print("bag:", NAME)
''')

# =========================================================================== 1. rosbag
md('''
## 1. rosbag 살펴보기

rosbag 은 ROS 토픽으로 오간 메시지를 **받은 시각과 함께 그대로** 저장한 것. 폴더 하나가 bag 하나이고,
안에 `metadata.yaml`(목록)과 `.db3`(메시지)가 있음.

조교가 차에서 녹화한 명령 (차 안에서 Ctrl+C 로 끝냄):

```bash
ros2 bag record --storage sqlite3 --output data/bags/run_train2 /flir_camera/image_raw /flir_camera/camera_info
```

녹화는 83초, 1.5 m/s 이하로 천천히. `real_ipm_lab` 은 그 원본(1.3 GB)을 씀. 여기서는 레포에 넣으려고 차선이 잘 보이는
15초만 잘라서 0.5초에 한 장만 남겼고, 사람 얼굴은 모자이크함. 아래 셀은 차에서 `ros2 bag info` 로 보는 것과 같은 내용.
''')
code('''
info = lab.bag_summary(BAG)
print("파일", info["files"], f"| 길이 {info['duration_s']:.1f} s")
pd.DataFrame(info["topics"])
''')
code('''
# 0.5초마다 한 장. 카메라 원본은 bayer 패턴이라 색을 복원해서 보여 줌
frames = lab.read_frames(BAG, every_s=0.5, topic=TOPIC)
print(len(frames), "장 |", frames[0]["bgr"].shape)
plt.figure(figsize=(10, 6.5)); plt.imshow(frames[0]["bgr"][..., ::-1]); plt.axis("off"); plt.title("camera image")
plt.show()
''')

# =========================================================================== 2. BEV
md('''
## 2. BEV 만들기

`real_ipm_lab` 에서 한 것과 **같은 경로**: `ost.yaml` 로 왜곡을 펴고, 카메라 자세(pitch, 높이)로 만든 homography 로
바닥을 위에서 본 BEV 로 폄 (`real.bev_from_camera`). 시뮬 BEV 와 같은 규격(300 x 380 px, 1 px = 1 cm, 앞 0.2~4.0 m)이라
같은 모델이 들어감. 카메라가 못 보는 구석은 회색.
''')
code('''
cam = lab.Camera(OST, cfg, PITCH_DEG, HEIGHT_M, OFFSET_X_M)
bevs = lab.make_bevs(frames, cam)
fig, axes = plt.subplots(2, 7, figsize=(14, 6.5))
for ax, k in zip(axes.flat, range(0, len(bevs), 2)):
    ax.imshow(bevs[k][..., ::-1]); ax.set_title(str(k)); ax.axis("off")
plt.tight_layout(); plt.show()
''')

# =========================================================================== 3. 라벨링
md('''
## 3. 라벨링

아래 셀을 실행하면 클릭 화면이 뜸. 십자 = 뒷바퀴 축, 점선 원 = 뒷바퀴 축에서 1 m.

1. **원이 좌우 테이프의 가운데와 만나는 곳을 클릭** (점이 원 위에 붙음, 드래그로 옮김)
2. 맞으면 **Enter**(승인) → 다음 프레임. 가운데를 모르겠으면(테이프가 안 보임, 흐림) **X**(제외)
3. **←, →** 로 앞뒤 이동, "다음 미작업" 으로 남은 프레임

키보드가 안 먹으면 화면을 한 번 클릭. 라벨은 누를 때마다 `out/<bag 이름>_labels.json` 에 저장되고,
셀을 다시 실행해도 남아 있음 (같은 bag 의 같은 프레임, 같은 카메라 자세일 때).
''')
code('''
labeler = lab.Labeler(frames, bevs, cfg, f"out/{NAME}_labels.json")
labeler.show()
''')
code('''
labeler.counts()      # 라벨링이 끝나면 실행해서 확인
''')

# =========================================================================== 4. 자동 라벨과 비교
md('''
## 4. 자동 라벨과 비교

`real_ipm_lab` 5장의 자동 라벨(노란 테이프 → 차선 둘 → 중심선 → 1 m 앞 점)을 같은 프레임에 돌려서 내가 찍은 점과 비교함.

- 자동 라벨이 **nan** 인 프레임: 차선 둘을 못 찾은 것. 사람은 찍을 수 있었는지 볼 것
- 둘 다 있는 프레임의 차이가 몇 cm 인지. 정의가 조금 달라서(자동 = 중심선 **따라** 1 m, 클릭 = 1 m **원** 위) 몇 cm 는 정상.
  크게 다르면 어느 쪽이 맞는지 BEV 를 보고 판단
''')
code('''
auto = lab.auto_labels(frames, cam, LANE_WIDTH_M)      # 차선 둘을 못 찾으면 nan
mine = np.array([l["waypoint_m"] if l["status"] == "accepted" else [np.nan, np.nan] for l in labeler.labels], float)
has_auto, has_mine = np.isfinite(auto).all(1), np.isfinite(mine).all(1)
both = has_auto & has_mine
gap_cm = np.where(both, np.hypot(*(auto - mine).T) * 100, np.nan)
print(f"자동 라벨 {has_auto.sum()} / {len(frames)}장 | 내가 승인 {has_mine.sum()}장 | 둘 다 있음 {both.sum()}장")
print("자동 라벨이 못 한 프레임:", np.flatnonzero(~has_auto).tolist())
if both.any():
    print(f"둘의 차이: 평균 {np.nanmean(gap_cm):.1f} cm, 최대 {np.nanmax(gap_cm):.1f} cm")
worst = [int(i) for i in np.argsort(-np.nan_to_num(gap_cm, nan=-1))[:4] if both[i]]   # 차이가 큰 순
fig, axes = plt.subplots(1, max(1, len(worst)), figsize=(12, 4.5), squeeze=False)
for ax, i in zip(axes[0], worst):
    img = bevs[i].copy()
    render.draw_points_bev(img, mine[i], cfg, (0, 200, 0), 7)      # 초록: 내가 찍은 점
    render.draw_points_bev(img, auto[i], cfg, (255, 128, 0), 5)    # 파랑: 자동 라벨
    ax.imshow(img[..., ::-1]); ax.set_title(f"frame {i}: {gap_cm[i]:.0f} cm apart"); ax.axis("off")
plt.suptitle("green: my click   blue: auto label (real_ipm_lab)"); plt.show()
''')

# =========================================================================== 5. 데이터셋
md('''
## 5. 데이터셋 (camsim 형식)

승인한 프레임만 모아서 `camsim_lab` 2장 "데이터셋 저장"과 **같은 폴더**를 만듦: `images/*.png` + `labels.csv`.

`labels.csv` 의 열은 같음(`file, x, y, theta, wp_x, wp_y`). 실차는 차가 맵 어디에 있는지 모르니
`x, y, theta` 는 비워 둠(`nan`). 학습에는 `wp_x, wp_y` (1 m 앞 점)만 씀.

6장에서 20% 를 검증용으로 떼므로 **승인한 프레임이 5장 이상**이어야 함.
''')
code('''
n = lab.write_dataset(labeler.bevs, labeler.labels, "out/week3_real", prefix=NAME)
print(n, "장 -> out/week3_real")
pd.read_csv("out/week3_real/labels.csv").head()
''')

# =========================================================================== 6. 학습
md('''
## 6. 학습

`camsim_lab` 3장과 **같은 함수**(`DiskDataset`, `train.train`)에 폴더 경로만 바꿔 넣음. 20% 는 검증용으로 떼어 둠.

30장으로는 모델이 장면을 외워 버림(train loss 는 내려가도 실력은 안 늚). 실제로 쓰려면 여러 주행·조명에서
수백~수천 장을 모아야 함. 오늘은 흐름 확인까지.
''')
code('''
train_ds = dataset.DiskDataset("out/week3_real", cfg, "train", val_frac=0.2)
val_ds = dataset.DiskDataset("out/week3_real", cfg, "val", val_frac=0.2)
print("train", len(train_ds), "/ val", len(val_ds))
STEPS, BATCH = 300, min(8, len(train_ds))
net, history = train.train(None, cfg, steps=STEPS, batch_size=BATCH, lr=1e-3, device=DEVICE,
                           dataset=train_ds, val_dataset=val_ds, log_every=25)
h = pd.DataFrame(history)
plt.plot(h["step"], h["loss"], label="train"); plt.plot(h["step"], h["val_loss"], label="val")
plt.yscale("log"); plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.show()
''')
code('''
pred = model.Predictor(net, cfg, DEVICE)
res = train.evaluate_dataset(pred, val_ds)
print(f"검증 {res['n']}장 평균 오차 {res['mean_m'] * 100:.1f} cm (최대 {res['max_m'] * 100:.1f} cm)")

fig, axes = plt.subplots(1, min(4, len(val_ds)), figsize=(12, 4.5), squeeze=False)
for ax, i in zip(axes[0], range(len(val_ds))):
    img = val_ds.load_image(i).copy()
    render.draw_points_bev(img, val_ds.wps[val_ds.idx[i]], cfg, (0, 200, 0), 7)   # 초록: 내가 찍은 정답
    render.draw_points_bev(img, pred.predict(img), cfg, (255, 0, 255), 5)         # 자홍: 모델 예측
    ax.imshow(img[..., ::-1]); ax.axis("off")
plt.suptitle("green: label   magenta: prediction"); plt.show()
''')

# =========================================================================== 정리
md('''
## 정리

- 녹화(rosbag) → 프레임 추출 → BEV (IPM) → 사람이 정답을 찍음 → camsim 형식 데이터셋 → 같은 코드로 학습
- 자동 라벨(`real_ipm_lab`)과 사람 라벨은 언제 갈리는가? 자동이 못 하는 프레임은 어떤 장면이었나?
- 30장이 아니라 3000장이 필요하다면 어떤 장면을 더 녹화해야 할까? (조명, 코너, 반대 방향, 다른 트랙)
''')
md('''
## 내 bag 으로 하기

1. 차에서 bag 을 0.5초에 한 장으로 줄임. 원본은 1분에 약 5 GB 라 그대로는 올리기 힘듦 (40 Hz 카메라면 20장에 한 장):
   `python3 camsim/scripts/decimate_bag.py data/bags/my_run data/bags/my_run_2hz 20`
2. 줄인 bag 폴더 안의 **`.db3` 파일 하나**(예: `my_run_2hz_0.db3`)와 그 카메라의 `ost.yaml` 을 노트북 컴퓨터로 가져와서
   (scp 나 USB) 왼쪽 **파일 창(📁)** 에 끌어다 놓음. 올린 파일은 `/content/` 에 생기고, 런타임이 끊기면 사라짐
3. 카메라 자세: 조교 차와 카메라 위치가 다르면 `real_ipm_lab` 을 내 bag(원본)으로 돌려 3장이 출력하는 pitch 와 높이를 받아 둠
4. 0장 파라미터 셀의 아래 줄들을 바꾸고 그 셀부터 다시 실행 (나머지 줄은 그대로 둠). 카메라 토픽 이름이 다르면 `TOPIC` 도 바꿈
   (1장 표에 bag 의 토픽이 나옴)

```python
BAG = "/content/my_run_2hz_0.db3"
OST = "/content/ost.yaml"
PITCH_DEG = ...      # real_ipm_lab 3장 출력
HEIGHT_M = ...
```
''')

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "week3_label_train.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
print("wrote", os.path.normpath(out), len(cells), "cells")
