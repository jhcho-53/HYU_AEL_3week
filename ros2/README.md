# 차에서 모델로 달리기

`real_ipm_lab` 9장에서 내보낸 모델로 차를 트랙에서 달리게 함. 노드 두 개를 씀. 둘 다 빌드 없이 레포 폴더에서 `python3` 로 바로 실행함.

- `ros2/waypoint_node.py` (모델 노드): 카메라 영상이 들어올 때마다 모델이 차선 가운데로 1 m 앞 점(waypoint)을 찍어서 `/waypoint` 로 보냄
- `ros2/drive_node.py` (주행 노드): `/waypoint` 를 받아서 그 점으로 가는 조향각을 계산하고 (2주차 시뮬레이터와 같은 pure pursuit), 정한 속도와 함께 `/drive` 로 보냄. 차량 스택이 이 값대로 모터와 조향을 움직임

```
카메라 → 모델 노드 → /waypoint → 주행 노드 → /drive → 차량 스택 → 모터
```

먼저 **A** 에서 녹화한 bag 으로 모델이 점을 제대로 찍는지 봄 (차는 안 움직임). 그다음 **B** 에서 트랙을 달림.

명령 블록 위의 **T1**~**T5** 는 그 명령을 붙여넣을 터미널.
터미널 열기 Ctrl+Alt+T, 붙여넣기 Ctrl+Shift+V (Ctrl+V 아님), 멈추기 Ctrl+C

## 0. 준비

### 0-1. 레포 받기 (차마다 한 번)

레포가 없으면 받음 (있으면 `cd ~/HYU_AEL_3week && git pull`):

```bash
cd ~
git clone https://github.com/jhcho-53/HYU_AEL_3week.git
```

### 0-2. 학습한 모델 가져오기 (새로 학습할 때마다)

코랩에서 `real_ipm_lab` 9장 셀을 실행하면, 학습한 모델이 구글 드라이브의 **내 드라이브 → `camsim_results` → `car_model.zip`** 으로 올라감 (다시 실행하면 덮어씀).
이 zip 을 차로 받아서 레포 폴더에 풂.

1. 전에 받은 zip 을 지움. 같은 이름이 남아 있으면 브라우저가 새 파일을 `car_model (1).zip` 으로 저장해서, 3번 명령이 옛 모델을 풀게 됨:

   ```bash
   rm -f ~/Downloads/car_model*.zip
   ```

2. 차 화면에서 Chromium 을 열고 drive.google.com 에 들어감. 내 드라이브 → `camsim_results` 에서 `car_model.zip` 을 오른쪽 클릭 → **다운로드**. `~/Downloads/` 에 저장됨
3. 레포 폴더에 풂. 전에 푼 `car_model` 폴더가 있으면 새 모델로 바뀜:

   ```bash
   cd ~/HYU_AEL_3week
   unzip -o ~/Downloads/car_model.zip
   ls car_model
   ```

   `model.onnx`, `checkpoint.json`, `H_i2g.npy`, `ost.yaml` 네 개가 나오면 됨
4. 여럿이 쓰는 차이므로 다 받았으면 Chromium 에서 구글 계정을 로그아웃함

## A. bag 으로 모델 확인 (차는 안 움직임)

| 터미널 | 쓰는 곳 |
|---|---|
| T1 | 모델 노드 |
| T2 | bag 재생 |
| T3 | `/waypoint` 숫자 보기 |
| T4 | 모델이 본 화면 보기 |

카메라는 켜지 않음. 카메라와 bag 이 같은 토픽(`/flir_camera/image_raw`)으로 영상을 내서, 둘 다 켜면 모델 노드에 두 영상이 섞여 들어감.

### A-1. 모델 노드 켜기

**T1** (끝까지 켜 둠):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
python3 ros2/waypoint_node.py --ros-args -p model_dir:=car_model
```

- `준비됨: 모델 car_model/model.onnx (CUDAExecutionProvider), 영상 /flir_camera/image_raw -> /waypoint` 줄이 나오면 됨.
  `CPUExecutionProvider` 면 GPU 를 못 쓰는 것 (느려도 돎)
- 영상이 들어오면 `첫 waypoint: x ... m, y ... m` 가 한 번 나옴

### A-2. bag 틀기

**T2** (스페이스바 = 일시정지. 화면을 멈추고 볼 때):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
ros2 bag play bag폴더
```

`bag폴더` 자리에 bag 폴더 경로를 넣을 것 (예: `data/bags/my_run`). 이 모델을 학습한 bag 과 같은 카메라 마운트로 녹화한 것이어야 BEV 가 맞음.

### A-3. 결과 보기

**T3** 모델이 내는 `/waypoint` 숫자 (Ctrl+C 로 끝냄):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
ros2 topic echo /waypoint
```

- `frame_id: rear_axle`, `point: x` 가 1 근처, `y` 가 커브 방향으로 바뀌면 정상 (+ 가 왼쪽)
- 1초에 몇 번 나오는지는 `ros2 topic hz /waypoint`

**T4** 모델이 본 BEV 와 찍은 점:

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view /waypoint_bev
```

- 하늘색 선 = 후륜축에서 1 m 원, 자홍 점 = 모델이 찍은 waypoint, 위 숫자 = 그 점의 x, y
- 두 테이프가 길고 나란하게 보여야 함. 짧게 휘어 보이면 카메라 마운트가 학습 bag 때와 달라진 것

### A-4. 끄기

T2 → T1 순서로 Ctrl+C. T3, T4 도 Ctrl+C

## B. 트랙에서 달리기

| 터미널 | 쓰는 곳 |
|---|---|
| T1 | 카메라 |
| T2 | 차량 스택 (조이스틱, 모터) |
| T3 | 모델 노드 |
| T4 | 주행 노드 |
| T5 | 확인 |

조이스틱은 이렇게 씀:
- **RB 를 누르고 있는 동안만** 모델로 달림. 떼면 바로 멈춤
- LB 를 누르면 언제든 스틱으로 직접 몰 수 있음 (모델보다 먼저 먹힘). LB 만 누르고 스틱을 안 움직이면 멈춤

시작 전에:
- A 에서 켠 터미널은 다 끔 (모델 노드가 두 개 켜지면 주행 노드가 안 달림)
- 작은 상자를 트랙 차선 가운데에 놓고, 차를 테이프와 나란하게 그 위에 올림. 바퀴는 바닥에 안 닿고, 카메라는 앞의 두 테이프를 봐야 함 (카메라가 차선을 못 보면 주행 노드가 안 달림). B-5 까지 이대로 둠

### B-1. 카메라 켜기

**T1** (끝까지 켜 둠):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
ros2 launch spinnaker_camera_driver driver_node.launch.py camera_type:=blackfly_s serial:="'카메라_serial'"
```

`카메라_serial` 자리에 차에 붙은 라벨의 카메라 serial 을 넣을 것 (따옴표는 그대로 둠)

### B-2. 차량 스택 켜기

**T2** (끝까지 켜 둠):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

LiDAR 오류(`Error connecting to Hokuyo`)는 무시함 (이 실습에서 안 씀)

### B-3. 모델 노드 켜기

**T3** (끝까지 켜 둠):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
python3 ros2/waypoint_node.py --ros-args -p model_dir:=car_model
```

`준비됨: ...` 다음에 `첫 waypoint: ...` 가 나오면 됨

### B-4. 주행 노드 켜기

**T4** (끝까지 켜 둠):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
python3 ros2/drive_node.py --ros-args -p speed:=0.5
```

- `speed:=0.5` 가 달리는 속도 (m/s). 처음엔 0.5 로 함
- 켜고 몇 초 안에 마지막 줄이 `대기: RB 를 누르고 있으면 달림` 이 되면 준비 끝
- 켜자마자 `정지: ...` 줄이 잠깐 나올 수 있음 (토픽을 찾는 중). 5초가 지나도 마지막 줄이 `정지: ...` 면 그 줄에 적힌 것을 확인함 (아래 문제 해결)

### B-5. 바퀴 띄우고 확인

차는 상자 위에 둔 채로 확인함.

**T5** 주행 노드가 보내는 값 (확인하고 Ctrl+C):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
ros2 topic echo /drive
```

- RB 를 누르고 있는 동안: T4 에 `주행: 0.5 m/s` 가 나오고, T5 의 `speed` 가 0.5, 바퀴가 앞으로 천천히 돎
- RB 를 떼면: `speed` 가 0.0 이 되고 바퀴가 멈춤
- 바퀴가 뒤로 돌거나 안 멈추면 LB 를 누른 채로 (스틱은 건드리지 않음) T4 를 Ctrl+C 로 끔. B-6 으로 넘어가지 않음

### B-6. 트랙에서 달리기

- 차를 상자에서 내려 트랙 차선 가운데에 테이프와 나란하게 놓음
- RB 를 누르고 있으면 달림. 테이프 밖으로 나갈 것 같으면 바로 RB 를 뗌
- 차를 출발 자리로 다시 가져올 때는 LB 를 누르고 스틱으로 몰거나 손으로 듦
- 잘 돌면 T4 를 Ctrl+C 로 끄고 `speed:=0.7`, `speed:=1.0` 처럼 올려서 다시 켬. 2.0 보다 크면 안 켜짐
- 모델이 본 화면을 같이 보려면 T5 에서 `ros2 run rqt_image_view rqt_image_view /waypoint_bev`

### B-7. 끄기

RB 에서 손을 뗀 뒤 T4 → T3 → T2 → T1 순서로 Ctrl+C. T5 도 Ctrl+C. T4 는 꺼지면서 마지막으로 속도 0 을 보냄

## 문제 해결

| 증상 | 해결 |
|---|---|
| `unzip: cannot find or open ~/Downloads/car_model.zip` | 다운로드가 안 됐거나 다른 이름으로 저장된 것. `ls ~/Downloads` 로 봄. `car_model (1).zip` 처럼 되어 있으면 0-2 의 1번부터 다시 |
| 차에서 구글 로그인이 안 됨 | 노트북으로 `car_model.zip` 을 받아서 USB 로 옮기고, 차의 `~/Downloads/` 에 넣은 뒤 0-2 의 3번부터 |
| `시작 못 함: ... 가 없음` | `car_model` 폴더에 파일 네 개가 다 있는지 `ls car_model` 로 확인. 9장 zip 을 다시 받아 풂 |
| `시작 못 함: ... checkpoint.json 과 다름` | `model.onnx` 가 깨졌거나 다른 모델의 것. 9장 zip 을 다시 받아 풂 |
| `No module named 'onnxruntime'` | `pip3 install onnxruntime` (CPU 로 돎) |
| `/waypoint` 가 안 나옴 | 영상(A 는 T2 bag, B 는 T1 카메라)이 돌고 있는지, `ros2 topic list` 에 `/flir_camera/image_raw` 가 있는지 볼 것 |
| `영상 처리 실패: 영상 ... 이 ost.yaml 의 ... 과 다름` | 카메라 해상도가 캘리브레이션과 다름. 1920x1200 으로 맞출 것 |
| T4 에 `정지: /waypoint 가 안 옴` | T1 카메라와 T3 모델 노드가 켜져 있는지 봄. `ros2 topic hz /waypoint` 에 숫자가 나와야 함 |
| T4 에 `정지: 모델이 1 m 앞이 아닌 엉뚱한 점을 냄` | 카메라가 트랙의 두 테이프를 보고 있는지 봄. 트랙 위인데도 계속 나오면 T5 에서 `ros2 run rqt_image_view rqt_image_view /waypoint_bev` 로 모델이 본 화면을 봄 |
| T4 에 `정지: /joy 가 안 옴` | 조이스틱 아무 버튼이나 눌러서 깨움. T2 차량 스택이 켜져 있는지 봄. `ros2 topic hz /joy` 가 약 20 Hz 여야 함 |
| RB 를 눌러도 T4 가 계속 `대기` | `ros2 topic echo /joy` 를 켜고 RB 를 눌러 봄. `buttons` 의 여섯 번째 숫자가 1 이 되어야 함 |
| T4 에 `정지: ... 를 내는 노드가 2개` | 같은 노드가 두 번 켜진 것. 열린 터미널을 보고 남는 것을 Ctrl+C 로 끔 (A 에서 켠 모델 노드, 속도를 바꾸기 전에 켠 주행 노드 등). 다 하나씩인데도 그러면 같은 와이파이의 다른 차와 토픽이 섞인 것. 조교를 부름 |
| T4 는 `주행` 인데 바퀴가 안 돎 | T2 화면에 VESC 오류가 있는지 봄. LB 로 스틱 주행이 되는지 봄 |
| `시작 못 함: speed=...` | `speed:=` 뒤에 0 보다 크고 2.0 이하인 숫자를 넣음 (예: 0.5) |
| 커브에서 테이프 밖으로 나감 | 속도를 0.5 로 낮춰 봄. 그래도 나가면 B-7 순서로 다 끄고 A 로 돌아가서 화면의 자홍 점이 차선 가운데를 찍는지 봄. 카메라를 건드렸으면 학습 bag 때와 BEV 가 달라진 것 |
