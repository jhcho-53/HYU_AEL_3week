"""정답 waypoint 와 학습용 pose 샘플링.

waypoint 는 하나. 직선 거리가 아니라 기준선을 따라간 호길이로 잡음. 직선 거리로 잡으면 코너에서
점이 안쪽을 파고들어 정답이 이상해짐.
"""
import numpy as np
from .config import Config
from .track import Track
from .render import to_vehicle


def nearest_index(track: Track, xy) -> int:
    d = track.center - np.asarray(xy)[:2]
    return int(np.argmin(d[:, 0] ** 2 + d[:, 1] ** 2))


def lateral_error(track: Track, xy) -> float:
    i = nearest_index(track, xy)
    return float(np.hypot(*(track.center[i] - np.asarray(xy)[:2])))


def waypoint_ahead(pose, track: Track, cfg: Config) -> np.ndarray:
    """pose 에서 기준선을 따라 ahead_m 앞의 점 (x, y), 차량 좌표계. 트랙 끝에서는 한 바퀴 돌아 이어짐."""
    i = nearest_index(track, pose[:2])
    s_t = (track.s[i] + cfg.waypoints.ahead_m) % track.length
    j = int(np.searchsorted(track.s, s_t)) % len(track.s)
    return to_vehicle(pose, track.center[j])


def sample_pose(track: Track, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """트랙 위 아무 데나 차를 놓음. 주행 없이 학습 데이터를 만들 수 있는 이유."""
    i = int(rng.integers(len(track.center)))
    corridor = float(track.left_m[i] + track.right_m[i])       # 그 지점의 실제 트랙 폭
    lat = rng.uniform(-1, 1) * cfg.sampling.lateral_frac * corridor
    dth = rng.uniform(-1, 1) * np.deg2rad(cfg.sampling.heading_deg)
    h = track.heading[i]
    n = np.array([-np.sin(h), np.cos(h)])
    xy = track.center[i] + lat * n
    return np.array([xy[0], xy[1], h + dth])


def body_corners(pose, cfg: Config) -> np.ndarray:
    """차체 사각형 네 모서리 (world). gym 의 collision_models.get_vertices 와 같은 규약으로
    pose 를 중심에 둔 length x width 사각형."""
    x, y, th = pose
    L, W = cfg.closed_loop.car_length_m / 2, cfg.closed_loop.car_width_m / 2
    local = np.array([[L, W], [L, -W], [-L, -W], [-L, W]])
    c, s = np.cos(th), np.sin(th)
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def signed_lateral(track: Track, xy):
    """기준선 대비 부호 있는 횡 오프셋(+ = 왼쪽)과 최근접 인덱스."""
    i = nearest_index(track, xy)
    n = np.array([-np.sin(track.heading[i]), np.cos(track.heading[i])])
    return float((np.asarray(xy)[:2] - track.center[i]) @ n), i


def crosses_tape(pose, track: Track, cfg: Config) -> bool:
    """차체 모서리 하나라도 테이프 안쪽 선을 넘으면 실격.

    벽 추종 트랙은 지점마다 폭이 다르므로 상수가 아니라 track.left_m / right_m 을 봄.
    """
    half_tape = cfg.lane.tape_width_m / 2
    for c in body_corners(pose, cfg):
        lat, i = signed_lateral(track, c)
        if lat > track.left_m[i] - half_tape or -lat > track.right_m[i] - half_tape:
            return True
    return False
