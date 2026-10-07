# 3주차 예제 bag

조교 차(F1TENTH, FLIR Blackfly S)로 2026-10-06 실습실 트랙을 천천히(최고 1.5 m/s) 달리며 녹화한 것.
`notebooks/week3_label_train.ipynb` 가 씀. `notebooks/real_ipm_lab.ipynb` 가 구글 드라이브에서 받는 원본 bag(1.3 GB)과 같은 주행.

| 파일 | 내용 |
|---|---|
| `run_train2_part1_2hz/` | rosbag2 (ROS 2 Humble, sqlite3). 83초 녹화 중 차선이 잘 보이는 15초를 잘라 0.5초에 한 장만 남김(영상 28장, GitHub 파일 크기 제한 100 MB 때문). 사람 얼굴은 모자이크 |
| `ost.yaml` | 렌즈 캘리브레이션 (K, D, 1920×1200). 체커보드 10×7, 25 mm, 재투영 오차 0.9 px. 원본 bag 의 `ost.yaml` 과 같음 |

bag 토픽: `/flir_camera/image_raw` (`sensor_msgs/Image`, `bayer_rggb8`, 1920×1200), `/flir_camera/camera_info`.

카메라 자세(pitch −4.25°, 높이 0.178 m)는 `real_ipm_lab` 3장이 원본 bag 에서 추정한 값이고, 노트북 파라미터 셀에 적혀 있음
(`camsim/week3_lab.py` 의 `SAMPLE_*` 와 같음).

녹화한 명령 (차에서):

```bash
ros2 bag record --storage sqlite3 --output data/bags/run_train2 /flir_camera/image_raw /flir_camera/camera_info
```

0.5초에 한 장으로 줄인 명령 (차에서, ROS 필요): `python3 camsim/scripts/decimate_bag.py data/bags/run_train2_part1 data/bags/run_train2_part1_2hz 20`
