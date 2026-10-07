# 실차 IPM 매뉴얼

실차 카메라 bag 으로 **카메라 homography(`H_i2g`)** 와 **실차 BEV 데이터셋**을 만들고, camsim 시뮬 학습과 연결하는 방법.
노트북은 [`notebooks/real_ipm_lab.ipynb`](../notebooks/real_ipm_lab.ipynb), 코드는 [`camsim/real.py`](../camsim/real.py).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhcho-53/HYU_AEL_3week/blob/main/notebooks/real_ipm_lab.ipynb)

## 1. 전체 흐름

```
실차 bag (bayer_rggb8)
  -> Bayer 디코딩 -> 왜곡 보정 (ost.yaml)
  -> pitch 추정 (정지 구간, 차선 평행) + 높이 (실측 차선 간격)       = 바닥 기준 extrinsic
  -> H_i2g.npy  ------------------------------------------------>  camsim_lab: cfg.camera.h_i2g_file
  -> 640x400 축소 -> render.ipm_bev -> BEV 380x300               = 모델 입력 (시뮬과 같은 규격)
  -> HSV 차선 검출 -> 중심선 -> 호길이 ahead_m 앞 waypoint        = 라벨
  -> out/real_dataset/ (images + labels.csv)  ------------------>  DiskDataset 으로 평가·학습
```

시뮬은 BEV 를 지오메트리에서 바로 그리고, 실차는 카메라 영상을 IPM 으로 펴서 같은 규격의 BEV 를 만든다.
둘이 같은 `H_i2g` 와 같은 BEV 규격을 쓰기 때문에 시뮬에서 학습한 모델을 실차 BEV 에 그대로 넣을 수 있다.

## 2. 준비물

### 2.1 bag 폴더
한 폴더에 아래 세 파일이 있어야 한다.

| 파일 | 내용 |
|---|---|
| `*.db3` | ROS 2 bag (sqlite3). 카메라 토픽 `/flir_camera/image_raw` (`sensor_msgs/Image`) |
| `metadata.yaml` | `ros2 bag record` 가 같이 만드는 파일 |
| `ost.yaml` | ROS `camera_calibration` 결과. **intrinsic(K, 왜곡 D)만** 들어 있음 |

토픽 이름이 다르면 `real.iter_images(..., topic=...)` 로 바꾼다. 인코딩은 `bayer_rggb8/bggr8/gbrg8/grbg8`, `rgb8`, `bgr8`, `mono8` 을 읽는다.

### 2.2 녹화할 때 지킬 것
- **출발 전 3초 이상 정지**: 곧은 차선 구간에서 두 테이프가 다 보이게 세워 둔 채 녹화를 시작한다.
  pitch 는 이 정지 구간 프레임으로만 추정한다 (커브에서는 "두 차선이 평행" 조건이 안 맞음).
- **카메라 마운트 고정**: 마운트가 바뀌면 extrinsic 이 바뀌므로 `H_i2g` 를 다시 만들어야 한다.
- **가능하면 `/tf_static` 도 녹화**: 지금 bag 엔 없어서 extrinsic 을 영상으로 추정한다.
  ```bash
  ros2 bag record /flir_camera/image_raw /flir_camera/camera_info /tf_static
  ```
- `camera_info` 는 지금 bag 에서 0 으로 비어 있다. 그래서 intrinsic 은 `ost.yaml` 에서 읽는다.

### 2.3 실측할 것

| 항목 | 필요 여부 | 재는 법 | 노트북 파라미터 |
|---|---|---|---|
| 차선 간격 | **필수** (아래 높이와 둘 중 하나) | 줄자로 두 테이프 **중심과 중심** 사이. 정지 구간 위치의 전방 0.5~2 m 에서 몇 군데 재서 평균 | `LANE_WIDTH_MEASURED` |
| 카메라 높이 | 선택 | 바닥에서 **렌즈 중심**까지 (로봇 윗면 아님). 넣으면 차선 간격보다 우선 | `CAM_HEIGHT_MEASURED` |
| 후륜축 -> 카메라 전방 거리 | 실차 제어에 쓸 때 | camsim 좌표 원점이 후륜축이라 waypoint 위치에 그대로 더해짐 | `OFFSET_X_M` |

단안 영상만으로는 미터 스케일(높이)을 알 수 없다. 지면 전체가 높이에 비례해 커지거나 작아질 뿐 영상이 똑같기 때문이다.
노트북은 차선 간격을 "카메라 높이의 몇 배"로 재므로 실측 간격 하나로 높이가 정해진다. 렌즈 중심 높이는 정확히 짚기 어려워서 차선 간격 쪽을 권장한다.

현재 bag 기준 값: 차선 간격 **0.80 m** (실측) -> 높이 **0.178 m**, pitch **-4.25 도** (살짝 위로 들림), 후륜축 거리 **0** (미측정).

## 3. 실행

### 3.1 코랩 (권장)
1. 위 배지로 노트북을 연다. GPU 런타임은 필요 없다.
2. **0장 파라미터 셀**에서 `LANE_WIDTH_MEASURED` 등을 확인하고 위에서부터 순서대로 실행한다.
3. 첫 실행 때 bag (약 1.3 GB) 을 `DRIVE_FOLDER_URL` 의 구글 드라이브 폴더에서 `out/real_bag/` 으로 받는다.
   다른 bag 을 쓰려면 드라이브 폴더 공유를 "링크가 있는 모든 사용자"로 바꾸고 그 주소를 넣는다.
4. 전체 실행은 다운로드 빼고 2~3분.

### 3.2 로컬
```bash
pip install rosbags gdown opencv-python-headless pyyaml pandas matplotlib imageio-ffmpeg
ln -s /path/to/bag_folder out/real_bag        # 이미 받은 bag 이 있으면 다운로드 대신 연결
jupyter notebook notebooks/real_ipm_lab.ipynb
```
torch 는 6장의 `DiskDataset` 확인과 7장 모델 평가에만 필요하다. 없으면 그 부분만 건너뛴다.

### 3.3 파라미터 (0장)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `cfg.waypoints.ahead_m` | 1.0 | 라벨 waypoint 의 전방 호길이 (m). **camsim_lab 과 같아야 함** |
| `BAG_DIR` | `out/real_bag` | bag 폴더 |
| `DRIVE_FOLDER_URL` | 3주차 bag 폴더 | `BAG_DIR` 이 비어 있을 때 받아 올 곳 |
| `LANE_WIDTH_MEASURED` | 0.80 | 실측 차선 간격 (m) |
| `CAM_HEIGHT_MEASURED` | None | 실측 카메라 높이 (m). 넣으면 우선 |
| `OFFSET_X_M` | 0.0 | 후륜축 -> 카메라 전방 거리 (m) |
| `SKIP_STATIC` | True | 정지 구간 프레임(거의 같은 그림)을 데이터셋에서 뺌 |

## 4. 장별 확인 항목
각 장 출력에서 아래가 맞는지 보고 넘어간다. 아니면 7절 문제 해결.

| 장 | 확인할 것 | 현재 bag 결과 |
|---|---|---|
| 1. bag | 정지 구간이 수십 프레임 이상 잡히는지 (움직임 그래프 초록 구간) | 556 프레임, 37.1 FPS, 정지 0~116 |
| 2. 디코딩 | 오른쪽 그림에서 테이프가 **노란색**인지 (하늘색이면 Bayer 패턴 문제) | 실측 화각 85.2 도 (config 가정 90 도) |
| 3. extrinsic | pitch 그래프에 뚜렷한 최소점. 오른쪽 격자의 하늘색 선이 테이프와 나란한지 | pitch -4.25 도, 높이 0.178 m |
| 4. BEV 비교 | 시뮬과 실차 BEV 의 회색(안 보이는 곳) 모양 일치율 | 99.9 % |
| 5. 라벨링 | 상태가 대부분 `two_lanes`, 초록 점이 두 테이프 사이 | 556/556 `two_lanes`, 튐 0 |
| 6. 저장 | `DiskDataset` 이 읽히는지, 미리보기 영상에서 빨간(skipped) 프레임 확인 | 439 장, 75 MB |

## 5. 결과물

```
out/real/
├── H_i2g.npy              camsim 해상도(640x400) 이미지 px -> 후륜축 기준 지면 m (3x3)
└── labels_preview.mp4     검수용. 전면 | BEV, 초록 = 라벨
out/real_dataset/
├── images/NNNNNN.png      실차 BEV 380x300 BGR. 파일명 = bag 프레임 번호
├── labels.csv             file, x, y, theta, wp_x, wp_y   (DiskDataset 포맷. x, y, theta 는 nan)
├── frames.csv             전체 프레임의 status, valid, jump, static, use, wp_x, wp_y, file
└── real_spec.json         bag, K, D, pitch, height, height_source, offset, lane width, ahead_m, bev, H_i2g
```

- **좌표계**: vehicle = 후륜축 원점, x 전방, y 좌측 (m). image = u 오른쪽, v 아래 (px). camsim `camera.py` 와 같다.
- **BEV 규격**: 전방 0.2~4.0 m, 좌우 ±1.5 m, 1 cm/px -> 380x300. 카메라가 못 보는 곳은 바닥색 (128,128,128).
- **라벨 정의**: 중심선 위에서 후륜축에 가장 가까운 점부터 중심선을 따라 `ahead_m` 간 점.
  camsim `gt.waypoint_ahead` 와 같은 정의다. 직선 거리로 잡으면 코너에서 점이 안쪽으로 파고든다.
- **valid**: 두 차선 검출 + waypoint 계산됨 + 앞뒤 3프레임 중앙값에서 8 cm 넘게 튀지 않음.
  `use` = valid 이고 (`SKIP_STATIC` 이면) 정지 구간이 아님. `labels.csv` 에는 `use` 프레임만 들어간다.

## 6. camsim_lab 과 연결

코랩은 노트북마다 런타임이 따로라 파일이 넘어가지 않는다. real_ipm_lab 7장이 드라이브 `MyDrive/camsim_results/real/` 에
`H_i2g.npy`, `real_spec.json`, `real_dataset.zip` 을 올린다.

### 6.1 시뮬 카메라를 실차 카메라로
camsim_lab 첫 셀(설치) 다음에 드라이브를 마운트하고 파일을 가져온다.
```python
from google.colab import drive; drive.mount("/content/drive")
!mkdir -p out/real && cp /content/drive/MyDrive/camsim_results/real/H_i2g.npy out/real/
!unzip -q -o /content/drive/MyDrive/camsim_results/real/real_dataset.zip -d out/
```
파라미터 셀에 real_ipm_lab 7장이 출력하는 줄을 붙인다 (현재 bag 기준 값).
```python
cfg.camera.h_i2g_file = "out/real/H_i2g.npy"
cfg.camera.height_m = 0.1783     # jitter_bev 증강이 기준 자세로 씀
cfg.camera.pitch_deg = -4.25
cfg.camera.offset_x_m = 0.0
cfg.lane.follow_walls = False    # 실습실 테이프 트랙처럼 일정 폭
cfg.lane.track_width_m = 0.80
cfg.waypoints.ahead_m = 1.0
```
`h_i2g_file` 이 들어가면 config 의 카메라 가정값(화각, 높이, pitch)보다 이 파일이 우선한다.
카메라·트랙 설정이 바뀌었으니 camsim_lab 2장이 시뮬 데이터를 알아서 다시 만든다.

### 6.2 학습한 모델의 실차 오차
camsim_lab 3장 학습 뒤:
```python
ds_real = dataset.DiskDataset("out/real_dataset", cfg, "all")
r_real = train.evaluate_dataset(pred, ds_real)
print(f"real: mean {r_real['mean_m']*100:.1f} cm, max {r_real['max_m']*100:.1f} cm")
```
real_ipm_lab 7장에서 `MODEL_PT` 에 드라이브의 `model.pt` 경로를 넣어도 같은 걸 잴 수 있다 (가장 틀린 4장도 보여줌).
시뮬만으로 학습한 모델은 실차 오차가 크게 나오는 게 정상이다. 그게 sim-to-real 갭이고, 3장 증강 과제 전후로 이 숫자를 비교한다.

### 6.3 실차 데이터를 학습에 섞을 때
- 연속 프레임은 거의 같은 그림이라 `DiskDataset` 의 무작위 9:1 분할을 쓰면 val 에 train 과 같은 장면이 들어가서 점수가 부풀려진다.
  **시간 구간이나 bag 단위로 나눈다.**
- 체크포인트의 `input_spec` (BEV 규격, waypoint 설정, 테이프 색, 모델 구조)이 다르면 `model.load` 가 거부한다.
  camsim_lab 에서 바꾼 값은 real_ipm_lab 파라미터 셀에도 똑같이 넣는다.

## 7. 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| bag 다운로드가 중간에 멈춤 | 구글 드라이브 대용량 파일 | 다운로드 셀 다시 실행. 계속 안 되면 직접 받아서 `BAG_DIR` 에 둠 |
| 테이프가 하늘색 | Bayer 패턴 이름 차이 | `real.decode_image` 를 쓰면 자동. 직접 바꿀 땐 ROS `rggb8` = OpenCV `BayerBG` |
| `no pitch candidate sees two lanes` | 정지 구간이 없거나, 정지 구간에 차선이 하나만 보이거나, 테이프 색이 HSV 범위 밖 | 정지 상태로 다시 녹화. 아니면 3장의 `static_idx` 를 두 차선이 곧게 보이는 프레임 번호로 직접 지정. 색이면 `real.HSV_LO/HSV_HI` 조정 |
| pitch 그래프 최소점이 없거나 여러 개 | 정지 구간이 커브거나 마스크에 잡음 | 위와 같이 곧은 구간 프레임으로 지정 |
| 3장 격자가 테이프와 안 나란함 | pitch 오추정, 또는 roll/yaw 가 0 이 아님 | 노트북 3장 "직접 바꿔 보기" 슬라이더로 확인. roll/yaw 가 크면 체커보드 + `cv2.solvePnP` 로 extrinsic 직접 측정 |
| 5장에서 `one_lane_*`, `no_lane` 많음 | 조명·반사로 테이프 검출 실패, 트랙 밖 주행 | 미리보기 영상으로 해당 프레임 확인. 색 문제면 HSV 범위 조정 |
| waypoint 가 nan | 중심선이 `ahead_m` 까지 안 보임 | `ahead_m` 을 줄임 (높이 0.18 m 카메라로는 2 m 넘기지 말 것) |
| `model.load` 가 거부 | `input_spec` 불일치 | 6.3 참고 |
| 코랩에서 영상이 안 뜸 | mp4v 코덱 | 노트북이 `viz.to_h264` 로 변환함. `imageio-ffmpeg` 설치 확인 |

## 8. 한계
- **평면 지면, roll = yaw = 0 가정.** 지면 위로 솟은 물체(벽, 의자, 사람)는 BEV 에서 길게 늘어진다.
- **주행 중 pitch 변화는 반영 안 함.** 가감속 때 차체가 숙거나 들리면 먼 곳 BEV 가 휜다. camsim 의 `jitter_bev` 증강이 이걸 흉내 낸다.
- **IPM 해상도는 멀수록 나빠진다.** 차선 검출은 전방 2.5 m 까지만 쓰고, waypoint 는 1 m 를 기본으로 둔다.
- **라벨은 차선 중앙이다.** 로봇이 실제로 달린 궤적이 아니다.
- 정확한 extrinsic 이 필요하면 로봇을 세워 두고 카메라 앞 바닥에 체커보드를 놓은 채 몇 초 녹화한 뒤 `cv2.solvePnP` 로
  높이·pitch·roll·yaw 를 한 번에 구한다. 그 결과로 `real.ground_homography` 대신 H 를 만들면 나머지 파이프라인은 그대로 쓴다.

## 9. 새 bag 추가
1. 2절 조건대로 녹화하고 `ost.yaml` 과 같은 폴더에 둔다 (카메라가 같으면 `ost.yaml` 재사용).
2. `BAG_DIR` (또는 `DRIVE_FOLDER_URL`) 과 `DATA_OUT` 을 bag 마다 다르게 해서 노트북을 다시 돌린다.
3. 카메라 마운트가 그대로면 pitch·높이가 이전과 비슷하게 나와야 한다. 크게 다르면 마운트가 움직였거나 추정이 틀린 것.
4. bag 별 데이터셋 폴더를 따로 두면 bag 단위 train/val 분할이 쉽다.
