"""Pure pursuit 조향. 속도는 부르는 쪽에서 정함."""
import numpy as np


def pure_pursuit(wp, wheelbase: float, steer_max: float) -> float:
    """예측 waypoint (x, y) 하나로 조향각. 그 점까지의 거리가 곧 lookahead."""
    x, y = np.asarray(wp, float).reshape(2)
    L2 = max(x * x + y * y, 1e-6)
    curvature = 2.0 * y / L2
    return float(np.clip(np.arctan(wheelbase * curvature), -steer_max, steer_max))
