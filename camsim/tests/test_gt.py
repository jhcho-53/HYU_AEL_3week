import numpy as np, pytest
from camsim import config, track, gt

@pytest.fixture(scope="module")
def ctx():
    cfg = config.load()
    return cfg, track.from_csv("examples/example_waypoints.csv", cfg)

def test_waypoint_is_ahead_on_straight(ctx):
    cfg, trk = ctx
    i = 10
    pose = np.array([*trk.center[i], trk.heading[i]])
    wp = gt.waypoint_ahead(pose, trk, cfg)
    assert wp.shape == (2,)
    assert abs(wp[0] - cfg.waypoints.ahead_m) < 0.3 and abs(wp[1]) < 0.3

def test_waypoint_is_forward_everywhere(ctx):
    cfg, trk = ctx
    for i in range(0, len(trk.center), 37):
        pose = np.array([*trk.center[i], trk.heading[i]])
        wp = gt.waypoint_ahead(pose, trk, cfg)
        assert wp[0] > 0 and np.hypot(*wp) > 0.5 * cfg.waypoints.ahead_m

def test_left_turn_has_positive_y(ctx):
    cfg, trk = ctx
    n = int(round(cfg.waypoints.ahead_m / cfg.lane.segment_len_m))
    dh = np.angle(np.exp(1j * (np.roll(trk.heading, -n) - trk.heading)))
    i = int(np.argmax(dh))              # ahead_m 구간에서 가장 센 좌회전
    pose = np.array([*trk.center[i], trk.heading[i]])
    assert gt.waypoint_ahead(pose, trk, cfg)[1] > 0.05

def test_wraps_at_track_end(ctx):
    cfg, trk = ctx
    i = len(trk.center) - 3
    pose = np.array([*trk.center[i], trk.heading[i]])
    wp = gt.waypoint_ahead(pose, trk, cfg)
    # 결승선 넘어도 앞쪽에 있어야 함. 리샘플 격자 때문에 정확히 ahead_m 은 아님
    assert np.all(np.isfinite(wp)) and wp[0] > cfg.waypoints.ahead_m * 0.6

def test_sample_pose_within_bounds(ctx):
    cfg, trk = ctx
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = gt.sample_pose(trk, cfg, rng)
        lat, i = gt.signed_lateral(trk, p[:2])          # 그 지점의 실제 트랙 폭 기준
        corridor = trk.left_m[i] + trk.right_m[i]
        assert abs(lat) <= cfg.sampling.lateral_frac * corridor + 0.05
        dth = np.angle(np.exp(1j * (p[2] - trk.heading[i])))
        assert abs(dth) <= np.deg2rad(cfg.sampling.heading_deg) + 0.05


def test_body_corners_are_rectangle_around_pose(ctx):
    cfg, trk = ctx
    c = gt.body_corners(np.array([1.0, 2.0, np.pi / 2]), cfg)
    assert c.shape == (4, 2)
    assert np.allclose(c.mean(0), [1.0, 2.0])
    L, W = cfg.closed_loop.car_length_m, cfg.closed_loop.car_width_m
    d = np.sort(np.hypot(*(c - [1.0, 2.0]).T))
    assert np.allclose(d, np.hypot(L / 2, W / 2))


def test_crosses_tape_uses_car_body_and_local_tape(ctx):
    """실격 = 차체 모서리가 그 지점의 테이프 안쪽 선을 넘음 (테이프 위치는 지점마다 다를 수 있음)."""
    cfg, trk = ctx
    i = 10
    h = trk.heading[i]
    n = np.array([-np.sin(h), np.cos(h)])
    margin = trk.left_m[i] - cfg.lane.tape_width_m / 2 - cfg.closed_loop.car_width_m / 2
    assert not gt.crosses_tape(np.array([*trk.center[i], h]), trk, cfg)
    assert not gt.crosses_tape(np.array([*(trk.center[i] + (margin - 0.05) * n), h]), trk, cfg)
    assert gt.crosses_tape(np.array([*(trk.center[i] + (margin + 0.05) * n), h]), trk, cfg)
    # 오른쪽도 대칭으로 동작
    margin_r = trk.right_m[i] - cfg.lane.tape_width_m / 2 - cfg.closed_loop.car_width_m / 2
    assert gt.crosses_tape(np.array([*(trk.center[i] - (margin_r + 0.05) * n), h]), trk, cfg)
