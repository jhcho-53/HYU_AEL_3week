# f1tenth_gym — camsim fork

[f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) 을 fork 해서 카메라 기반 waypoint 실습 `camsim/` 을 얹은 레포다.
시뮬레이터(`gym/f110_gym/`)는 업스트림 그대로고, 카메라 렌더링·데이터셋·학습·폐루프 검증은 전부 `camsim/` 에 있다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhcho-53/HYU_AEL_3week/blob/main/notebooks/camsim_lab.ipynb)

배지를 누르면 코랩에서 실습 노트북이 열린다. 첫 셀이 레포를 clone 하고 의존성을 설치하므로
로컬에 아무것도 깔 필요가 없다. 자세한 내용은 [camsim/README.md](camsim/README.md).

실차 카메라 bag 으로 `H_i2g` 와 실차 BEV 데이터셋을 만드는 노트북은 따로 있다 (GPU·gym 불필요).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhcho-53/HYU_AEL_3week/blob/main/notebooks/real_ipm_lab.ipynb)

## 구성

    camsim/            렌더러, 데이터셋, 모델, 학습, 폐루프, 드라이브 전달, 실차 bag 처리(real.py)
    notebooks/         실습 노트북. camsim_lab (시뮬 학습), real_ipm_lab (실차 데이터)
                       원본은 camsim/scripts/build_notebook.py, build_real_notebook.py
    gym/f110_gym/      업스트림 시뮬레이터 (수정 없음)
    examples/          맵과 중심선. 트랙 지오메트리의 출처

## 업스트림 인용

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
