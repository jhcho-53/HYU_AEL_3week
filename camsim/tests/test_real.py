"""real.py 검증. 실차 bag 대신 camsim 렌더러로 그린 합성 카메라 영상을 씀.
pitch / 높이 / 트랙을 알고 그렸으니, 실차 파이프라인이 그걸 되찾는지 보면 됨."""
import copy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from camsim import config, camera, track, render, gt, real

CSV = "examples/example_waypoints.csv"
PITCH, HEIGHT, WIDTH = -4.0, 0.20, 0.8      # 실차 추정값과 비슷한 자세 (살짝 위로 들림)


@pytest.fixture(scope="module")
def ctx():
    cfg = config.load()
    cfg.lane.follow_walls = False           # 일정 폭. 실습실 테이프 트랙 방식
    cfg.lane.track_width_m = WIDTH
    cfg.camera.pitch_deg = PITCH
    cfg.camera.height_m = HEIGHT
    trk = track.from_csv(CSV, cfg)
    return cfg, trk, camera.build(cfg)[0]


def _straight_index(trk, span=60):
    """앞으로 span 점(3 m) 동안 헤딩이 가장 안 변하는 지점."""
    h = np.unwrap(trk.heading)
    n = len(h) - span
    return int(np.argmin(np.abs(h[span:span + n] - h[:n])))


def _curve_index(trk, span=30, target_deg=25.0):
    """앞으로 1.5 m 동안 헤딩이 target 정도 꺾이는 지점."""
    h = np.unwrap(trk.heading)
    n = len(h) - span
    return int(np.argmin(np.abs(np.abs(np.rad2deg(h[span:span + n] - h[:n])) - target_deg)))


def _shot(cfg, trk, H_g2i, i):
    pose = np.array([*trk.center[i], trk.heading[i]])
    return pose, render.render(pose, trk.quads, None, H_g2i, cfg)


def test_ground_homography_matches_camera_module(ctx):
    cfg, _, H_g2i = ctx
    cfg2 = copy.deepcopy(cfg)
    cfg2.camera.offset_x_m = 0.12
    H_ref = camera.build(cfg2)[0]
    H = real.ground_homography(camera.intrinsics(cfg2), PITCH, HEIGHT, 0.12)
    pts = np.array([[0.5, 0.3], [1.0, -0.4], [2.5, 0.0]])
    assert np.allclose(camera.project(H, pts), camera.project(H_ref, pts), atol=1e-6)


def test_scaled_intrinsics_project_to_scaled_pixels():
    K = np.array([[1044.3, 0, 967.3], [0, 1040.7, 592.0], [0, 0, 1]])
    Ks = real.scale_intrinsics(K, (1920, 1200), (640, 400))
    pts = np.array([[0.8, 0.2], [1.5, -0.3]])
    full = camera.project(real.ground_homography(K, -4.25, 0.18), pts)
    small = camera.project(real.ground_homography(Ks, -4.25, 0.18), pts)
    assert np.allclose(small, full / 3.0, atol=1e-6)


def test_camsim_h_i2g_inverts_ground_projection(ctx):
    cfg, _, _ = ctx
    K = np.array([[1044.3, 0, 967.3], [0, 1040.7, 592.0], [0, 0, 1]])
    H_i2g = real.camsim_h_i2g(K, (1920, 1200), cfg, -4.25, 0.18)
    Ks = real.scale_intrinsics(K, (1920, 1200), real.camsim_size(cfg))
    pts = np.array([[0.6, 0.1], [2.0, -0.5]])
    uv = camera.project(real.ground_homography(Ks, -4.25, 0.18), pts)
    assert np.allclose(camera.project(H_i2g, uv), pts, atol=1e-6)


def test_horizon_row_matches_far_ground_projection(ctx):
    cfg, _, H_g2i = ctx
    v_far = camera.project(H_g2i, np.array([[1e4, 0.0]]))[0, 1]
    assert abs(real.horizon_row(camera.intrinsics(cfg), PITCH) - v_far) < 0.5


def test_bayer_rggb_is_decoded_with_red_in_bgr_channel_2():
    h, w = 8, 8
    mosaic = np.zeros((h, w), np.uint8)
    mosaic[0::2, 0::2] = 220               # R  (rggb: 짝수 행 짝수 열)
    mosaic[0::2, 1::2] = 110               # G
    mosaic[1::2, 0::2] = 110               # G
    mosaic[1::2, 1::2] = 10                # B
    msg = SimpleNamespace(data=mosaic.tobytes(), height=h, width=w, step=w, encoding="bayer_rggb8")
    b, g, r = real.decode_image(msg)[2:-2, 2:-2].reshape(-1, 3).mean(0)
    assert r > g > b


def test_estimate_pitch_recovers_synthetic_pitch_and_lane_gap(ctx):
    cfg, trk, H_g2i = ctx
    K = camera.intrinsics(cfg)
    _, img = _shot(cfg, trk, H_g2i, _straight_index(trk))
    mask = real.lane_mask(img, real.horizon_row(K, PITCH) + 5)
    pitch, cand, costs, gaps = real.estimate_pitch([mask], K)
    assert abs(pitch - PITCH) <= 0.5
    gap_h = gaps[0, list(cand).index(pitch)]
    assert abs(gap_h * HEIGHT - WIDTH) < 0.05 * WIDTH


@pytest.mark.parametrize("where, tol_m", [("straight", 0.03), ("curve", 0.06)])
def test_label_frame_matches_gt_waypoint(ctx, where, tol_m):
    cfg, trk, H_g2i = ctx
    K = camera.intrinsics(cfg)
    i = _straight_index(trk) if where == "straight" else _curve_index(trk)
    pose, img = _shot(cfg, trk, H_g2i, i)
    lab = real.label_frame(img, H_g2i, WIDTH, cfg.waypoints.ahead_m, real.horizon_row(K, PITCH) + 5)
    assert lab["status"] == "two_lanes"
    assert np.linalg.norm(lab["wp"] - gt.waypoint_ahead(pose, trk, cfg)) < tol_m


def test_waypoint_along_is_arc_length_not_straight_distance():
    xs = np.linspace(0.3, 2.0, 200)
    mids = np.column_stack([xs, 0.3 * xs ** 2])            # 왼쪽으로 휘는 중심선
    wp, _ = real.waypoint_along(mids, 1.0)
    C = np.column_stack([np.linspace(0, wp[0], 2000), 0.3 * np.linspace(0, wp[0], 2000) ** 2])
    arc = np.linalg.norm(np.diff(C, axis=0), axis=1).sum()
    assert abs(arc - 1.0) < 0.01 and np.hypot(*wp) < 1.0


def test_waypoint_along_is_nan_when_centerline_too_short():
    mids = np.column_stack([np.linspace(0.3, 0.6, 50), np.zeros(50)])
    wp, _ = real.waypoint_along(mids, 1.0)
    assert np.isnan(wp).all()


def test_flag_jumps_marks_only_the_outlier():
    wps = np.column_stack([np.full(11, 1.0), np.linspace(0, 0.05, 11)])
    wps[5, 1] += 0.3
    wps[8] = np.nan
    assert np.flatnonzero(real.flag_jumps(wps)).tolist() == [5]


def test_label_writer_matches_disk_dataset_format(tmp_path):
    with real.LabelWriter(tmp_path) as w:
        w.add(np.zeros((380, 300, 3), np.uint8), [1.0, -0.1])
        w.add(np.zeros((380, 300, 3), np.uint8), [0.9, 0.2])
    rows = (tmp_path / "labels.csv").read_text().splitlines()
    assert rows[0].split(",") == real.LABEL_HEADER and len(rows) == 3
    assert (tmp_path / "images" / "000001.png").is_file()
    dataset = pytest.importorskip("camsim.dataset")      # torch 가 있어야 import 됨
    assert real.LABEL_HEADER == dataset.LABEL_HEADER
    files, poses, wps = dataset.read_labels(str(tmp_path))
    assert files == ["000000.png", "000001.png"] and np.isnan(poses).all()
    assert np.allclose(wps, [[1.0, -0.1], [0.9, 0.2]])


def test_bev_from_camera_matches_sim_bev(ctx):
    """실차 경로(원본 해상도 영상 -> 축소 -> ipm_bev)가 시뮬 BEV 와 같은 테이프 위치를 내는지."""
    cfg, trk, _ = ctx
    K_full = real.scale_intrinsics(camera.intrinsics(cfg), real.camsim_size(cfg), (1920, 1200))
    H_full = real.ground_homography(K_full, PITCH, HEIGHT)
    cfg_full = copy.deepcopy(cfg)
    cfg_full.camera.image_width, cfg_full.camera.image_height = 1920, 1200
    pose = np.array([*trk.center[_straight_index(trk)], trk.heading[_straight_index(trk)]])
    img_full = render.render(pose, trk.quads, None, H_full, cfg_full)

    H_i2g = real.camsim_h_i2g(K_full, (1920, 1200), cfg, PITCH, HEIGHT)
    bev_real = real.bev_from_camera(img_full, H_i2g, cfg)
    bev_sim = render.render_bev(pose, trk.quads, cfg, render.bev_visibility_mask(np.linalg.inv(H_i2g), cfg))
    assert bev_real.shape == bev_sim.shape
    # test_render.test_ipm_bev_agrees_with_render_bev_near 와 같은 기준: 1.5 m 안쪽에서 IPM 테이프가
    # (5x5 로 불린) 참 테이프 위에 있는지. 축소(INTER_AREA)로 테이프 가장자리가 섞여서 반대 방향 비율은 낮게 나옴
    near = slice(int(round((cfg.bev.x_range_m[1] - 1.5) / cfg.bev.resolution_m)), None)
    a = real.lane_mask(bev_real)[near] > 0
    s = cv2.dilate((real.lane_mask(bev_sim)[near] > 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert a.sum() * cfg.bev.resolution_m ** 2 > 0.01
    assert (a & s).sum() / a.sum() > 0.9
