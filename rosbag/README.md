# rosbag 녹화 (차에서)

차 카메라 영상을 rosbag 으로 녹화하고, 0.5초에 한 장으로 줄여서
`notebooks/real_ipm_lab.ipynb` 맨 아래 "내 bag 으로 하기" 에 씀.

명령 블록 위의 **T1**·**T2**·**T3** 는 그 명령을 붙여넣을 터미널.
터미널 열기 Ctrl+Alt+T, 붙여넣기 Ctrl+Shift+V (Ctrl+V 아님), 멈추기 Ctrl+C

| 터미널 | 쓰는 곳 |
|---|---|
| T1 카메라 | 1번에서 켜고 끝까지 둠 |
| T2 차량 스택 | 2번에서 켜고 끝까지 둠 |
| T3 그때그때 | 레포 받기, 확인, 녹화, 줄이기 |

## 0. 레포 받기 (차마다 한 번)

**T3**

```bash
cd ~
git clone https://github.com/jhcho-53/HYU_AEL_3week.git
```

`already exists` 가 나오면 이미 받은 것. `cd ~/HYU_AEL_3week && git pull` 로 최신으로 맞춤

## 1. 카메라 켜기

**T1** (끝까지 켜 둠):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
ros2 launch spinnaker_camera_driver driver_node.launch.py camera_type:=blackfly_s serial:="'카메라_serial'"
```

`카메라_serial` 자리에 차에 붙은 라벨의 카메라 serial 을 넣을 것 (따옴표는 그대로 둠)

**T3** 영상이 들어오는지 확인 (숫자가 나오면 Ctrl+C):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
ros2 topic hz /flir_camera/image_raw
```

## 2. 차량 스택 켜기

**T2** (끝까지 켜 둠):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

LiDAR 오류(`Error connecting to Hokuyo`)는 무시함 (이 실습에서 안 씀)

**T3** 조이스틱 신호 확인 (약 20 Hz 가 나오면 Ctrl+C):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
ros2 topic hz /joy
```

## 3. 녹화

**T3** (끝낼 때 Ctrl+C 한 번):

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
mkdir -p data/bags
ros2 bag record --storage sqlite3 \
  --qos-profile-overrides-path rosbag/recording_qos.yaml \
  --output data/bags/my_run /flir_camera/image_raw /flir_camera/camera_info
```

- `Subscribed to topic` 줄이 2개 나오면 녹화가 시작된 것
- 녹화 중에 **곧은 구간에서 두 테이프가 다 보이는 장면**이 한 번은 있게 함. 출발 전에 곧은 곳에서 몇 초 세워 두면 가장 좋음
  (`real_ipm_lab` 3장에서 이 장면의 테이프에 점 4개를 찍어 카메라 높이와 기울기를 잼)
- 테이프 간격(두 테이프 **가운데와 가운데** 사이)을 줄자로 재서 적어 둠 (노트북 `LANE_WIDTH_MEASURED`)
- LB 를 누른 채 스틱으로 천천히 트랙을 돎. 왼쪽 스틱 위아래가 속도, 오른쪽 스틱 좌우가 조향. 버튼에서 손을 떼면 멈춤
- LB 만 눌렀는데 바퀴가 한쪽으로 꺾이면 바로 손을 떼고 손 들기
- 1분에 약 5 GB 가 쌓임. 한 번에 30~60초만
- 다 돌았으면 Ctrl+C 를 한 번 누르고 프롬프트가 다시 나올 때까지 기다림
- `already exists` 가 나오면 같은 이름이 이미 있는 것. `my_run` 을 `my_run2` 처럼 바꿔서 다시 (아래 명령의 이름도 같이 바꿈)
- `rosbag/recording_qos.yaml` 은 카메라 영상을 best effort 로 받게 하는 설정

## 4. 확인

**T3**

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
ros2 bag info data/bags/my_run
sync
```

- 두 토픽의 `Count` 가 0 이 아니면 정상
- `sync` 가 끝나기 전에는 차 전원을 끄거나 배터리·어댑터를 바꾸지 말 것 (bag 이 깨질 수 있음)

## 5. 0.5초에 한 장으로 줄이기

1분에 약 5 GB 라 그대로는 드라이브에 올리기 어려움. 차에서 0.5초에 한 장만 남긴 bag 을 새로 만듦.
노트북 5-2 클릭 라벨링은 원래 0.5초에 한 장만 써서 그대로임. 대신 자동 라벨로 만드는 데이터셋도 그만큼(약 20분의 1) 줄어듦.

**T3**

```bash
cd ~/HYU_AEL_3week
source /opt/ros/humble/setup.bash
python3 camsim/scripts/decimate_bag.py data/bags/my_run data/bags/my_run_2hz 20
ros2 bag info data/bags/my_run_2hz
sync
```

- `20` 은 몇 장에 한 장을 남길지. 카메라가 초당 약 40장이라 20 이면 0.5초에 한 장
- `my_run_2hz` 폴더에 그 카메라의 `ost.yaml`(1주차 결과)을 넣고 노트북 컴퓨터로 가져가서,
  구글 드라이브 `내 드라이브/week3/` 에 올림 (`real_ipm_lab` 맨 아래 "내 bag 으로 하기")


## 끄기

T2, T1 순서로 Ctrl+C

## 문제 해결

| 증상 | 해결 |
|---|---|
| 녹화 화면에 `Subscribed to topic` 줄이 안 나옴 | T1 카메라가 꺼져 있음. T1 을 다시 켜고, 녹화를 Ctrl+C 로 끈 뒤 3번을 다시 |
| `recording_qos.yaml` 을 못 찾는다는 오류 | `cd ~/HYU_AEL_3week` 를 빠뜨렸거나 레포가 옛것. `git pull` 후 다시 |
