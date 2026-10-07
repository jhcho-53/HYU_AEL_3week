# camsim — 합성 카메라 waypoint 파이프라인

`f1tenth_gym` 은 카메라를 못 그린다. 그런데 바닥 테이프 트랙은 전부 평면이라, 캘리브레이션 행렬
하나만 있으면 "이 위치에서 카메라로 보면 이렇게 보인다"를 계산해서 가짜 영상을 그릴 수 있다.
이 패키지가 그 렌더러와 학습, gym 폐루프 검증을 담는다.

## 실행: 코랩에서 노트북 하나

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhcho-53/HYU_AEL_3week/blob/main/notebooks/camsim_lab.ipynb)

학생용 경로는 `notebooks/camsim_lab.ipynb` 하나다. 위 배지를 누르면 코랩에서 열리고,
첫 셀이 레포를 clone 하고 의존성을 설치한다. **로컬에 설치할 게 없고 브라우저만 있으면 된다.**
데이터 생성, 학습, 폐루프 주행, 영상 재생까지 전부 코랩 안에서 돈다.

0장 설치·파라미터 → 1장 카메라와 트랙 → 2장 GT 와 데이터셋 → 3장 학습(train/val loss 실시간)
→ 4장 폐루프(횡오차 곡선, 주행 영상, 맵 위 경로) → 5장 드라이브 보관 + 젯슨용 ONNX.

각 장 첫 셀의 파라미터를 바꾸고 그 장을 다시 실행하면서 뭐가 달라지는지 본다.
`config.yaml` 은 기본값이고, 노트북 파라미터 셀이 그 위에 덮어쓴다.

노트북 원본은 `camsim/scripts/build_notebook.py` 다. 노트북을 고칠 일이 있으면 그 파일을 고치고
다시 생성한다.

    python camsim/scripts/build_notebook.py

## 학생이 손대는 곳

**노트북 0장의 "파라미터" 셀 하나다.** 카메라 높이·각도·화각, 테이프 폭, waypoint 거리, 모델 구조, 속도가
전부 거기 모여 있다. 그리고 3장의 `my_augment` 함수.

`camsim/config.yaml` 은 기본값 파일이고 **코랩에서는 직접 고치지 않는다.** 파라미터 셀이 그 위에
덮어쓰기 때문에 고쳐도 효과가 없고, 레포 파일을 건드리면 다음 실행 때 `git pull` 이 충돌한다.
(실수로 고쳤으면 `!git checkout -- .` 로 되돌리면 된다.)

캘리브레이션이 끝나면 파라미터 셀에 `cfg.camera.h_i2g_file = "..."` 로 `.npy` 경로를 적는다.
그러면 카메라 가정값은 전부 무시된다.

## 모델 입력은 BEV

앞에서 본 영상이 아니라 위에서 내려다본 BEV 를 모델에 넣는다. 이유는 실차와 규격을 맞추기 위해서다.

    시뮬 : render.render_bev(pose, quads, cfg, mask) 가 테이프를 top-down 으로 바로 그린다
    실차 : 카메라 -> undistort -> render.ipm_bev(cam, H_i2g, cfg)

`mask = render.bev_visibility_mask(H_g2i, cfg)` 로 카메라가 못 보는 영역(코앞 사각지대, 화각 밖)을
바닥색으로 덮으면 실차 IPM 출력과 같은 모양이 된다. 그래서 `Predictor.predict(bev)` 하나를 양쪽이
공유할 수 있다. 실차용 편의 함수로 `Predictor.predict_camera(cam, H_i2g)` 도 있다.

원근 렌더(`render.render`)는 눈으로 확인하거나 IPM 과 비교할 때만 쓴다.
BEV 범위와 해상도는 `config.yaml` 의 `bev:` 섹션 하나로 정한다. 기본은 전방 0.2~4 m, 좌우 ±1.5 m,
1 cm/px → 380×300 이다. 테이프 5 cm 가 5 px 이고, 카메라가 1.5 m 에서 주는 해상도(전후 3.5 cm/px)보다 촘촘하다.

## 실차 데이터: `real.py` 와 `notebooks/real_ipm_lab.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhcho-53/HYU_AEL_3week/blob/main/notebooks/real_ipm_lab.ipynb)

실차 카메라 bag (ROS 2, `bayer_rggb8`) 에서 위 "실차" 경로의 재료 두 가지를 만든다. GPU 도 gym 도 필요 없고,
bag (약 1.3 GB) 은 첫 실행 때 구글 드라이브에서 받는다. 노트북 원본은 `camsim/scripts/build_real_notebook.py`.
녹화 조건, 실측할 것, 장별 확인 항목, 문제 해결은 [실차 IPM 매뉴얼](../docs/real_ipm_manual.md) 에 있다.

    out/real/H_i2g.npy      camsim 해상도(640x400) 이미지 px -> 후륜축 기준 지면 m. cfg.camera.h_i2g_file 로 넣음
    out/real_dataset/       실차 BEV + labels.csv (DiskDataset 포맷 그대로). x, y, theta 는 nan (world pose 없음)

bag 에는 extrinsic 이 없다 (`camera_info` 는 0 으로 비어 있고 `/tf` 도 녹화 안 됨). `ost.yaml` 은 intrinsic 뿐이다.
그래서 pitch 는 정지 구간에서 위에서 본 두 차선이 평행해지는 값으로, 높이는 실측 차선 간격(테이프 중심 간) 하나로 정한다.
roll, yaw 는 0 가정. 지금 bag 기준 pitch -4.25 도(살짝 위로 들림), 높이 0.178 m (차선 간격 0.80 m 실측).

라벨은 `gt.waypoint_ahead` 와 같은 정의다. HSV 로 노란 테이프를 골라 지면에 펴고, 두 차선의 중점으로 중심선을
만들고, 후륜축에 가장 가까운 점에서 호길이 `ahead_m` 앞의 점을 잡는다. `test_real.py` 가 camsim 렌더러로 그린
합성 영상에서 pitch 와 waypoint 를 되찾는지 확인한다.

camsim_lab 에서 실차 카메라로 시뮬 데이터를 만들려면 파라미터 셀에 노트북 8장이 출력하는 줄들을 붙인다
(`h_i2g_file`, `height_m`, `pitch_deg`, `follow_walls = False`, `track_width_m`). 학습한 모델의 실차 오차는
`train.evaluate_dataset(pred, dataset.DiskDataset("out/real_dataset", cfg, "all"))`. 연속 프레임은 거의 같은
그림이라 실차 데이터를 학습에 섞을 땐 무작위 9:1 말고 시간 구간이나 bag 단위로 나눌 것.

## waypoint 는 1개, 거리는 config 로

모델은 전방 `waypoints.ahead_m` 지점 하나의 (x, y) 를 낸다. 기본 `1.0` 이고 단위는 미터,
기준선을 따라간 호길이다. 그 점이 곧 pure pursuit 의 목표점이라 lookahead 설정이 따로 없다.

거리를 바꾸면 라벨이 바뀐다. 노트북 2장이 `spec.json` 을 보고 데이터를 다시 만들고, 옛 `model.pt` 는
`input_spec` 검사에서 거부되니 조용히 섞일 일은 없다.

1 m 인 이유: 카메라 높이 0.2 m, 정면 장착 기준으로 PV → IPM → BEV 복구 오차를 재 보면 1.5 m 까지는
테이프 위치가 0.5 cm 안이고, 2 m 를 넘으면 코너에서 7 cm, 3 m 는 직선에서도 7 cm 다.
2 m 안쪽에서 고를 것. 더 멀리 보려면 카메라를 올려야 한다 (0.35 m + 15° 숙이면 3 m 에서 0.4 cm).

## 모델

```
BEV (3, 380, 300) BGR 0~1
  (x - mean) / std                    ImageNet 정규화. 모델 안에 있어서 젯슨도 같은 전처리가 됨
ResNet-18 백본 (ImageNet 사전학습)     → (512, 12, 10)
1×1 conv 512→32 · BN · ReLU           → (32, 12, 10)
flatten 3840 → FC 256 → ReLU → FC 2   → (x, y) m
```

- **사전학습 백본** 인 이유: 시뮬 BEV 는 깨끗한 노란 줄뿐이라 처음부터 배운 모델은 실차의 조명·그림자·바닥 질감에
  무너지기 쉽다. 시뮬 오차는 작은 모델과 비슷하고, 차이는 노트북 3장 열화 표에서 드러난다.
- **pooling 없는 head**: 답이 점의 *위치* 라 공간 정보를 평균으로 지우면 안 된다. 또 `AdaptiveAvgPool` 은
  feature map 이 출력 크기로 안 나눠떨어지면 legacy ONNX export 가 실패해서 TensorRT 로 못 넘어간다.
- **BGR**: ImageNet 가중치는 RGB 로 학습됐다. conv1 의 입력 채널 순서를 뒤집어 BGR 을 그대로 받게 해서
  채널 교환 연산이 ONNX 에 안 남는다. `test_resnet_takes_bgr_like_imagenet_takes_rgb` 가 이걸 검증한다.

`model.arch: small` 이면 같은 head 에 작은 CNN 백본 (37만 → head 포함 123만 파라미터). 학습이 훨씬 빠르고
젯슨이 느리면 이걸 쓴다. 구조가 체크포인트 `input_spec` 에 들어가서 다른 구조로는 로드가 거부된다.

## 데이터셋

`dataset.generate_dataset` 이 `out/dataset/images/NNNNNN.png` (BEV) 와 `labels.csv`
(file, x, y, theta, wp_x, wp_y) 를 만든다. 같이 저장되는 `spec.json` 이 어떤 설정으로 만든
데이터인지 기록해서, 설정이 바뀌면 노트북 2장이 알아서 다시 만든다. 저장되는 건 증강 없는 원본이고, train/val 은 9:1 로
결정적으로 나뉜다.

증강은 로딩 때 `DiskDataset(..., augment_fn=fn)` 의 `fn(bev, rng) -> bev` 하나로 넣는다 (기본 None).
같은 데이터로 증강만 바꿔 가며 비교할 수 있게 이렇게 나눠 놨다.

`camsim/augment.py` 에 OpenCV 기반 증강이 들어 있다. 세기는 config 의 `augment:` 에서 오고 기본값은
전부 "변화 없음"이다.

| 분류 | 함수 |
|---|---|
| 기하 | `jitter_bev` (pitch 변화로 BEV 가 휨), `ipm_blur` (먼 곳일수록 뭉개짐), `erase_patches` (테이프 마모) |
| 조명 | `brightness_contrast`, `gamma`, `hsv_shift`, `illumination`, `shadow` |
| 센서 | `blur`, `noise`, `jpeg` |
| 조합 | `example_augment` |

`jitter_bev` 외에는 실제 카메라 물리의 근사라, 실차 영상을 찍어 본 뒤 다시 설계하는 게 맞다.
`train.evaluate(..., degrade_fn=fn)` 으로 열화된 입력에 대한 강건성을 잴 수 있다.
sim-to-real 증강 설계가 노트북 3장 과제다. 실차 IPM 결과를 같은 `labels.csv` 포맷으로 저장하면
그대로 학습된다.

데이터는 코랩 로컬 디스크에 만들고 드라이브에는 zip 하나로 옮길 것. 드라이브는 작은 파일이 많으면
견딜 수 없이 느리다.

## 정답 경로 기준 (`waypoints.line`)

- `center` — 좌우 테이프의 중간선. 실차 HSV+IPM 자동 라벨링과 같은 기준이라 기본값이다.
- `racing` — `examples/example_waypoints.csv` 의 레이싱 라인. 코너 안쪽을 파고들어 더 짧다 (156 m vs 163 m).

어느 쪽이든 테이프(모델이 보는 것)는 똑같고 정답만 달라진다. 두 기준으로 학습해서 랩타임과 안정성을
비교하는 게 노트북 2장 실습이다. `Track.left_m` / `right_m` 이 그 기준선에서 좌우 테이프까지의
거리이고, pose 샘플링과 실격 판정이 이 값을 쓴다.

## 테이프 배치

`lane.follow_walls: true` 면 맵 PNG 의 벽까지 거리를 재서 `wall_margin_m` 안쪽에 테이프를 놓는다.
트랙 모양이 맵과 일치하는 대신 폭이 구간마다 다르다. `false` 면 중심선에서 `track_width_m/2` 로
일정하게 놓는다. 실습실 테이프 트랙과 같은 방식이라, 실차 트랙 치수가 정해지면 이쪽으로 바꾼다.

## 실격 규칙

실차 트랙은 벽 없이 테이프가 경계다. 시뮬도 같은 규칙을 쓴다. 차체
(`closed_loop.car_length_m` x `car_width_m`) 네 모서리 중 하나라도 테이프 안쪽 선을 넘으면
`reason="tape_crossed"` 로 종료한다. 완주(`lap`)는 테이프를 한 번도 안 넘고 한 바퀴 돈 것이다.
gym 벽 충돌(`collision`)은 이 맵에서 벽이 멀어서 거의 안 난다.

## 좌표계

world = gym 맵 (m). vehicle = 후륜축 원점, x 전방, y 좌측. image = OpenCV (u 우, v 아래).
`H_g2i` 는 ground (x,y,1) -> image. pitch 가 0이면 지평선이 이미지 세로 중앙에 온다.

## Colab → Google Drive → Jetson (TensorRT)

젯슨은 `model.onnx` 를 받아서 TensorRT 엔진을 **젯슨에서 직접** 만든다. 엔진은 GPU·TRT 버전마다 달라서
코랩에서 만든 건 못 쓴다. 학습된 가중치는 ONNX 안에 다 있어서 젯슨엔 PyTorch 가 필요 없다.

1. Colab 배지로 노트북을 열고 GPU 런타임에서 0~5장을 순서대로 실행한다.
2. 5장 셀이 `model.onnx` 를 만들고 onnxruntime 으로 PyTorch 결과와 같은지 확인한 뒤
   `MyDrive/camsim_results/` 로 올린다. 같은 폴더에 SHA-256·설정·git commit 을 담은 `checkpoint.json`,
   재학습용 `model.pt` 도 쓴다. 데이터셋까지 보관하려면 셀의 `SAVE_DATASET = True`.
3. 드라이브 웹에서 **`model.onnx` 파일**의 공유를 "링크가 있는 모든 사용자: 뷰어"로 바꾸고 링크를 복사한다.
4. 젯슨에서 (`DRIVE_FILE_LINK`, `SHA256_FROM_COLAB` 는 바꿀 것):

   ```bash
   python3 -m venv "$HOME/camsim-transfer-venv"
   "$HOME/camsim-transfer-venv/bin/python" -m pip install gdown
   "$HOME/camsim-transfer-venv/bin/python" -m gdown 'DRIVE_FILE_LINK' -O "$HOME/model.onnx"
   printf '%s  %s\n' 'SHA256_FROM_COLAB' "$HOME/model.onnx" | sha256sum -c -
   /usr/src/tensorrt/bin/trtexec --onnx="$HOME/model.onnx" --saveEngine="$HOME/model.engine" --fp16
   ```

   엔진 입력 `bev` 는 (1, 3, 380, 300) BGR 0~1 float, 출력 `wp` 는 (1, 2) = (x, y) / `waypoints.norm_m`.
   BEV 는 노트북과 같은 `config.yaml` 로 `render.ipm_bev` 를 거친 것이어야 한다.

## 코랩 주의

- f110_gym 의 오래된 `setup.py` 는 NumPy 상한을 요구하므로, 노트북은 Colab 기본 NumPy/PyTorch 를
  유지하고 f110_gym 을 `pip install --no-deps -e .` 로 설치한다.
- 설치 중에 numpy 가 바뀌면 런타임을 한 번 재시작해야 한다. 안 하면 `numpy.dtype size changed` 가 난다.
- 주행 영상이 셀에 안 뜨면 코덱 문제다. OpenCV 의 mp4v 는 브라우저가 못 읽으므로 `viz.to_h264()` 로
  변환해서 띄운다 (노트북은 이미 그렇게 한다).
- `pyglet` import 에러가 나면 `!apt-get install -y libgl1` 후 재시도. `f110_env` 가 모듈 최상단에서
  `from pyglet import gl` 을 해서, 창을 안 띄워도 GL 라이브러리가 있어야 한다.
- 노트북 첫 코드 셀의 `REPO_URL` 기본값은 조교 fork 다. 다른 fork 를 쓰면 그 줄만 바꾸면 된다.
  clone 이 실패하면 `%cd f1tenth_gym` 부터 전부 깨지므로 주소를 먼저 확인할 것.
- 세션이 끊기면 로컬 VM 파일이 사라진다. `model.onnx`, `model.pt` 는 5장에서 드라이브로 옮긴다.

## 로컬에서 테스트만 돌리려면

gym 없이도 도는 테스트가 대부분이라, 렌더러나 트랙 쪽을 고쳤을 때는 로컬에서 바로 확인할 수 있다.

    pip install opencv-python-headless pyyaml pillow pytest imageio-ffmpeg
    python -m pytest camsim/tests -q --ignore=camsim/tests/test_closed_loop.py \
        --ignore=camsim/tests/test_dataset_model.py --ignore=camsim/tests/test_disk_dataset.py \
        --ignore=camsim/tests/test_train.py

전부 돌리려면 torch 와 f110_gym 이 필요하다. f110_gym 의 오래된 의존성 선언은 최신 Colab 과
맞지 않고, gym 0.19 는 setup.py 에 오타가 있어서 `camsim/scripts/install_gym019.sh` 로 따로 설치해야 한다.
그냥 코랩에서 돌리는 게 빠르다.
