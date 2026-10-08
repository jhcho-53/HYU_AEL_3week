# 차에서 모델 돌리기: 카메라 → 모델 → `/waypoint`

`real_ipm_lab` 9장이 내보낸 모델로, 카메라 영상이 들어올 때마다 후륜축에서 1 m 앞 점을 `/waypoint`
(`geometry_msgs/PointStamped`, `rear_axle` 기준, x 전방, y 왼쪽) 로 냄. BEV 는 노트북과 같은 경로로 만듦
(`ost.yaml` 로 왜곡 보정 → `H_i2g.npy` 로 IPM). 빌드 없이 `python3` 로 바로 실행함.

이 노드는 `/waypoint` 만 냄. 모터 쪽(`/drive`)은 건드리지 않으므로 차는 움직이지 않음.

명령 블록 위의 **T1**~**T4** 는 그 명령을 붙여넣을 터미널.
터미널 열기 Ctrl+Alt+T, 붙여넣기 Ctrl+Shift+V (Ctrl+V 아님), 멈추기 Ctrl+C

## 0. 준비 (차마다 한 번)

레포가 없으면 받음 (있으면 `cd ~/HYU_AEL_3week && git pull`):

```bash
cd ~
git clone https://github.com/jhcho-53/HYU_AEL_3week.git
```

`real_ipm_lab` 9장이 드라이브에 올린 `car_model.zip` 을 받아서 레포 폴더에 풂. 풀면 `~/HYU_AEL_3week/car_model/` 에
`model.onnx`, `checkpoint.json`, `H_i2g.npy`, `ost.yaml` 네 개가 있어야 함:

```bash
cd ~/HYU_AEL_3week
unzip -o ~/Downloads/car_model.zip
ls car_model
```

## 1. 모델 노드 켜기

**T1** (끝까지 켜 둠):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
python3 ros2/waypoint_node.py --ros-args -p model_dir:=car_model
```

- `준비됨: 모델 car_model/model.onnx (CUDAExecutionProvider), 영상 /flir_camera/image_raw -> /waypoint` 줄이 나오면 됨.
  `CPUExecutionProvider` 면 GPU 를 못 쓰는 것 (느려도 돎)
- 영상이 들어오면 `첫 waypoint: x ... m, y ... m` 가 한 번 나옴

## 2. 영상 넣기

카메라 대신 녹화해 둔 bag 을 틀어도 되고, 카메라를 켜도 됨. 둘 중 하나만.

**T2** bag 재생 (스페이스바 = 일시정지. 화면을 멈추고 볼 때):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
ros2 bag play bag폴더
```

`bag폴더` 자리에 bag 폴더 경로를 넣을 것 (예: `data/bags/my_run`). 이 모델을 학습한 bag 과 같은 카메라 마운트로 녹화한 것이어야 BEV 가 맞음.

**T2** 또는 카메라:

```bash
source /opt/ros/humble/setup.bash
ros2 launch spinnaker_camera_driver driver_node.launch.py camera_type:=blackfly_s serial:="'카메라_serial'"
```

`카메라_serial` 자리에 차에 붙은 라벨의 카메라 serial 을 넣을 것 (따옴표는 그대로 둠)

## 3. 결과 보기

**T3** 모델이 내는 `/waypoint` 숫자 (Ctrl+C 로 끝냄):

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /waypoint
```

- `frame_id: rear_axle`, `point: x` 가 1 근처, `y` 가 커브 방향으로 바뀌면 정상 (+ 가 왼쪽)
- 몇 번 나오는지는 `ros2 topic hz /waypoint`

**T4** 모델이 본 BEV 와 찍은 점:

```bash
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view /waypoint_bev
```

- 하늘색 선 = 후륜축에서 1 m 원, 자홍 점 = 모델이 찍은 waypoint, 위 숫자 = 그 점의 x, y
- 두 테이프가 길고 나란하게 보여야 함. 짧게 휘어 보이면 카메라 마운트가 학습 bag 때와 달라진 것

## 끄기

T2 → T1 순서로 Ctrl+C. T3, T4 도 Ctrl+C

## 문제 해결

| 증상 | 해결 |
|---|---|
| `시작 못 함: ... 가 없음` | `car_model` 폴더에 파일 네 개가 다 있는지 `ls car_model` 로 확인. 9장 zip 을 다시 받아 풂 |
| `시작 못 함: ... checkpoint.json 과 다름` | `model.onnx` 가 깨졌거나 다른 모델의 것. 9장 zip 을 다시 받아 풂 |
| `No module named 'onnxruntime'` | `pip3 install onnxruntime` (CPU 로 돎) |
| `/waypoint` 가 안 나옴 | T2 가 돌고 있는지, `ros2 topic list` 에 `/flir_camera/image_raw` 가 있는지 볼 것 |
| `영상 처리 실패: 영상 ... 이 ost.yaml 의 ... 과 다름` | 카메라 해상도가 캘리브레이션과 다름. 1920x1200 으로 맞출 것 |
