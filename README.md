# HYU AEL 3주차 — 카메라 기반 waypoint 실습

바닥 테이프 트랙을 카메라로 보고 전방 waypoint 하나를 예측하는 모델을 만드는 실습 레포다.
[f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) 시뮬레이터 위에 두 가지를 얹었다.

- **시뮬 학습** (`camsim`): 시뮬에서 BEV 를 그려 데이터셋을 만들고, 모델을 학습하고, gym 안에서 폐루프로 검증한다.
- **실차 IPM** (`camsim/real.py`): 실차 카메라 bag 으로 카메라 homography 와 실차 BEV 데이터셋을 만든다.
  시뮬 카메라를 실차와 같게 맞추고, 학습한 모델의 실차 오차를 잰다.

모델 입력은 위에서 내려다본 BEV 다. 시뮬은 BEV 를 바로 그리고 실차는 카메라 영상을 IPM 으로 펴서
같은 규격(380x300, 1 cm/px)을 만들기 때문에 같은 모델이 양쪽에 들어간다.

## 노트북

| 노트북 | 내용 | 런타임 | 열기 |
|---|---|---|---|
| `notebooks/real_ipm_lab.ipynb` | 실차 bag -> 왜곡 보정 -> extrinsic 추정 -> `H_i2g.npy` + 실차 BEV 데이터셋 | CPU (GPU·gym 불필요) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhcho-53/HYU_AEL_3week/blob/main/notebooks/real_ipm_lab.ipynb) |
| `notebooks/camsim_lab.ipynb` | 시뮬 BEV 데이터 -> ResNet-18 학습 -> 폐루프 -> 젯슨용 ONNX | T4 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhcho-53/HYU_AEL_3week/blob/main/notebooks/camsim_lab.ipynb) |

배지를 누르면 코랩에서 열린다. 첫 셀이 이 레포를 clone 하고 필요한 것을 설치하므로 로컬에 깔 것이 없다.

## 진행 순서

1. **real_ipm_lab**: 실차 bag 으로 `H_i2g.npy` 와 `real_dataset` 을 만들고 드라이브 `MyDrive/camsim_results/real/` 에 올린다.
   실측값은 차선 간격 하나면 된다 (현재 bag: 0.80 m -> 카메라 높이 0.178 m, pitch -4.25 도).
   7장에서 같은 데이터로 camsim 학습 코드가 도는지 짧게 테스트한다.
2. **camsim_lab**: 파라미터 셀에 real_ipm_lab 8장이 출력한 줄을 붙여 시뮬 카메라를 실차 카메라로 바꾸고 학습한다.
3. 학습한 모델을 `real_dataset` 으로 평가해 sim-to-real 갭을 재고, camsim_lab 3장 증강 과제로 줄인다.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/real_ipm_manual.md](docs/real_ipm_manual.md) | **실차 IPM 매뉴얼.** 녹화 조건, 실측할 것, 실행, 장별 확인 항목, 결과물 포맷, camsim_lab 연결, 문제 해결 |
| [camsim/README.md](camsim/README.md) | camsim 설계. BEV 규격, waypoint 정의, 모델 구조, 증강, 실격 규칙, 젯슨 전달 |

## 구성

    camsim/            렌더러, 데이터셋, 모델, 학습, 폐루프, 드라이브 전달, 실차 bag 처리(real.py)
    camsim/tests/      pytest. test_real.py 는 합성 영상으로 실차 파이프라인을 검증
    notebooks/         실습 노트북. 원본은 camsim/scripts/build_notebook.py, build_real_notebook.py
    docs/              실차 IPM 매뉴얼
    gym/f110_gym/      업스트림 시뮬레이터 (수정 없음)
    examples/          맵과 중심선. 트랙 지오메트리의 출처

노트북을 고칠 때는 `.ipynb` 가 아니라 생성 스크립트를 고치고 다시 생성한다.

    python camsim/scripts/build_notebook.py
    python camsim/scripts/build_real_notebook.py

## 테스트

gym 과 torch 없이 도는 테스트가 대부분이다.

    pip install opencv-python-headless pyyaml pillow pytest imageio-ffmpeg rosbags
    python -m pytest camsim/tests -q --ignore=camsim/tests/test_closed_loop.py \
        --ignore=camsim/tests/test_dataset_model.py --ignore=camsim/tests/test_disk_dataset.py \
        --ignore=camsim/tests/test_train.py

전부 돌리려면 torch 와 f110_gym 이 필요하다 ([camsim/README.md](camsim/README.md) 의 "로컬에서 테스트만 돌리려면").

## 업스트림 인용

시뮬레이터는 [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) (MIT, [LICENSE](LICENSE)) 이다.

```
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O'Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```
