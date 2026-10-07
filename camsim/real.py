"""실차 bag -> camsim 규격 (H_i2g, BEV, labels.csv).

시뮬에서 학습한 모델을 실차에 그대로 쓰려면 실차 쪽이 세 가지를 맞춰야 함
  카메라 : 실측 K(ost.yaml) + extrinsic(높이, pitch) 로 만든 H_i2g. config 의 camera.h_i2g_file 로 넘김
  입력   : undistort -> camsim 해상도로 축소 -> render.ipm_bev
  라벨   : labels.csv 의 wp_x, wp_y. 기준은 gt.waypoint_ahead 와 같은 "중심선 따라 ahead_m 호길이"

bag 에는 extrinsic 이 없음 (camera_info 는 0 으로 비어 있고 /tf 도 녹화 안 됨). 그래서
  pitch  : 정지 구간에서 위에서 본 두 차선이 평행해지는 값을 찾음
  height : 단안 영상만으론 스케일을 모름. 실측 차선 간격(테이프 중심 간)이나 실측 높이로 정함
  roll, yaw 는 0 으로 가정.

좌표계는 camera.py 와 같음. vehicle = 후륜축 원점, x 전방, y 좌측. image = u 우, v 아래.
torch 없이 돌아감 (dataset.py 는 torch 를 import 해서 라벨 헤더를 여기 따로 둠).
"""
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import Config
from . import camera, render

IMAGE_TOPIC = "/flir_camera/image_raw"
LABEL_HEADER = ["file", "x", "y", "theta", "wp_x", "wp_y"]   # dataset.LABEL_HEADER 와 같아야 함 (테스트가 확인)

# 노란 테이프 HSV 범위. 먼 테이프는 밝은데 채도가 낮아서(S 80~110, V 255) S 하한을 낮게 둠
HSV_LO, HSV_HI = (18, 50, 120), (36, 255, 255)


# ---- bag --------------------------------------------------------------------------

def open_bag(bag_dir):
    from rosbags.highlevel import AnyReader          # ROS 설치 없이 rosbag2 를 읽는 순수 파이썬 라이브러리
    from rosbags.typesys import Stores, get_typestore
    return AnyReader([Path(bag_dir)], default_typestore=get_typestore(Stores.ROS2_HUMBLE))


def iter_images(bag_dir, topic: str = IMAGE_TOPIC, step: int = 1):
    """(index, 녹화 시각 ns, sensor_msgs/Image) 를 순서대로."""
    with open_bag(bag_dir) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise ValueError(f"{topic} not in bag; topics: {[c.topic for c in reader.connections]}")
        for i, (conn, t, raw) in enumerate(reader.messages(connections=conns)):
            if i % step == 0:
                yield i, t, reader.deserialize(raw, conn.msgtype)


def load_images(bag_dir, indices, topic: str = IMAGE_TOPIC) -> dict:
    wanted, out = set(int(i) for i in indices), {}
    for i, _, msg in iter_images(bag_dir, topic):
        if i in wanted:
            out[i] = msg
            if len(out) == len(wanted):
                break
    return out


# ROS 인코딩 이름 -> OpenCV 상수. OpenCV 는 Bayer 패턴을 둘째 행 기준으로 불러서 rggb8 이 BayerBG 임 (cv_bridge 도 같음).
# BayerRG 를 쓰면 R/B 가 뒤바뀌어 노란 테이프가 하늘색이 됨
BAYER_TO_BGR = {
    "bayer_rggb8": cv2.COLOR_BayerBG2BGR,
    "bayer_bggr8": cv2.COLOR_BayerRG2BGR,
    "bayer_gbrg8": cv2.COLOR_BayerGR2BGR,
    "bayer_grbg8": cv2.COLOR_BayerGB2BGR,
}


def decode_image(msg) -> np.ndarray:
    """sensor_msgs/Image -> BGR uint8."""
    buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
    if msg.encoding in BAYER_TO_BGR:
        return cv2.cvtColor(buf, BAYER_TO_BGR[msg.encoding])
    if msg.encoding == "rgb8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
    if msg.encoding == "bgr8":
        return buf.reshape(msg.height, msg.width, 3).copy()
    if msg.encoding == "mono8":
        return cv2.cvtColor(buf, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported encoding: {msg.encoding}")


def scan_motion(bag_dir, topic: str = IMAGE_TOPIC):
    """프레임마다 (녹화 시각 ns, 직전 프레임과의 평균 밝기 차). raw 모자이크를 8배 줄여서 비교."""
    stamps, motion, prev = [], [], None
    for _, t, msg in iter_images(bag_dir, topic):
        small = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[::8, ::8].astype(np.float32)
        motion.append(np.nan if prev is None else float(np.abs(small - prev).mean()))
        stamps.append(t)
        prev = small
    return np.array(stamps), np.array(motion)


def static_end(motion: np.ndarray, thresh: float = 4.0) -> int:
    """처음으로 움직이기 시작한 프레임. 그 앞이 정지 구간 (pitch 추정에 씀)."""
    moving = np.nan_to_num(motion) > thresh
    return int(np.argmax(moving)) if moving.any() else len(motion)


# ---- intrinsic ----------------------------------------------------------------------

def load_ost(path):
    """ROS camera_calibration 결과(ost.yaml) -> (K 3x3, D, (w, h)). intrinsic 만 들어 있음."""
    c = yaml.safe_load(Path(path).read_text())
    K = np.array(c["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(c["distortion_coefficients"]["data"], dtype=np.float64)
    return K, D, (int(c["image_width"]), int(c["image_height"]))


def undistort_maps(K, D, size):
    """왜곡 보정 remap 테이블. 새 K 를 원래 K 로 둬서 보정 후 영상도 같은 K 를 씀."""
    return cv2.initUndistortRectifyMap(K, D, None, K, size, cv2.CV_32FC1)


def scale_intrinsics(K, src_size, dst_size) -> np.ndarray:
    """영상을 src_size -> dst_size 로 resize 했을 때의 K."""
    sx, sy = dst_size[0] / src_size[0], dst_size[1] / src_size[1]
    return np.diag([sx, sy, 1.0]) @ K


def camsim_size(cfg: Config):
    return cfg.camera.image_width, cfg.camera.image_height


def to_camsim(img_undist: np.ndarray, cfg: Config) -> np.ndarray:
    """왜곡 보정된 원본 해상도 영상 -> camsim 카메라 해상도 (기본 640x400)."""
    h, w = img_undist.shape[:2]
    cw, ch = camsim_size(cfg)
    if abs(w / h - cw / ch) > 0.01:
        raise ValueError(f"aspect ratio mismatch: image {w}x{h} vs camsim {cw}x{ch}")
    return cv2.resize(img_undist, (cw, ch), interpolation=cv2.INTER_AREA)


# ---- extrinsic -> homography ----------------------------------------------------------

def ground_homography(K, pitch_deg: float, height_m: float, offset_x_m: float = 0.0) -> np.ndarray:
    """vehicle 지면 (x, y, 1) -> 이 K 의 이미지 (u, v, w). camera.extrinsics 와 같은 규약 (pitch + = 아래)."""
    th = np.deg2rad(pitch_deg)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(th), -np.sin(th)],
                   [0.0, np.sin(th), np.cos(th)]])
    R = Rx @ camera._R_VC
    t = -R @ np.array([offset_x_m, 0.0, height_m])
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return H / np.linalg.norm(H)        # camera._assumed_h_g2i 와 같은 정규화. project() 는 스케일 무관


def horizon_row(K, pitch_deg: float) -> float:
    """무한히 먼 지면이 찍히는 행."""
    return K[1, 2] - K[1, 1] * np.tan(np.deg2rad(pitch_deg))


def camsim_h_i2g(K_full, full_size, cfg: Config, pitch_deg: float, height_m: float,
                 offset_x_m: float = 0.0) -> np.ndarray:
    """camsim 해상도 이미지 px -> vehicle 지면 m. cfg.camera.h_i2g_file 로 넣을 행렬."""
    K = scale_intrinsics(K_full, full_size, camsim_size(cfg))
    H_i2g = np.linalg.inv(ground_homography(K, pitch_deg, height_m, offset_x_m))
    return H_i2g / np.linalg.norm(H_i2g)


def bev_from_camera(img_undist_full: np.ndarray, H_i2g: np.ndarray, cfg: Config) -> np.ndarray:
    """실차 모델 입력. 원본 해상도 보정 영상 -> camsim 해상도 -> render.ipm_bev (380x300 기본)."""
    return render.ipm_bev(to_camsim(img_undist_full, cfg), H_i2g, cfg)


# ---- 지면 격자 (라벨링용 BEV) -------------------------------------------------------------
# 픽셀 규약은 render.bev_pixels 와 같음 (위 = 전방, 왼쪽 = +y). 범위만 따로 받음

def grid_matrix(x_range, y_range, res) -> np.ndarray:
    """지면 (x, y, 1) -> 격자 픽셀 (u, v, 1)."""
    return np.array([[0.0, -1.0 / res, y_range[1] / res],
                     [-1.0 / res, 0.0, x_range[1] / res],
                     [0.0, 0.0, 1.0]])


def grid_size(x_range, y_range, res):
    return int(round((y_range[1] - y_range[0]) / res)), int(round((x_range[1] - x_range[0]) / res))


def warp_to_grid(img, H_g2i, x_range, y_range, res, interp=cv2.INTER_LINEAR) -> np.ndarray:
    """이미지를 지면 격자로 폄. H_g2i 는 이 이미지 해상도의 homography."""
    H_img2grid = grid_matrix(x_range, y_range, res) @ np.linalg.inv(H_g2i)
    return cv2.warpPerspective(img, H_img2grid, grid_size(x_range, y_range, res), flags=interp)


def grid_to_ground(rows, cols, x_range, y_range, res) -> np.ndarray:
    return np.column_stack([x_range[1] - np.asarray(rows) * res, y_range[1] - np.asarray(cols) * res])


def ground_to_grid(pts, x_range, y_range, res) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([(y_range[1] - p[:, 1]) / res, (x_range[1] - p[:, 0]) / res])


def lane_mask(img_bgr: np.ndarray, v_min: float = 0) -> np.ndarray:
    """노란 테이프 마스크. v_min 위쪽(지평선 근처, 벽·물체)은 버림."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, HSV_LO, HSV_HI)
    m[:max(0, int(v_min))] = 0
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


# ---- pitch 추정 -------------------------------------------------------------------------

def lane_gap_stats(mask, K, pitch_deg, x_range=(2.5, 7.0), y_range=(-5.0, 5.0), res=0.025):
    """높이 1 로 펴서 행마다 두 차선 간격을 잼. (변동계수 std/mean, 평균 간격 [높이 단위]).

    pitch 가 맞으면 두 차선이 평행이라 간격이 일정함. 범위도 높이 단위 (h=0.18 m 면 0.45~1.3 m).
    두 덩어리로 갈린 행이 60 % 안 되면 판정 불가로 (nan, nan).
    """
    g = warp_to_grid(mask, ground_homography(K, pitch_deg, 1.0), x_range, y_range, res, cv2.INTER_NEAREST)
    gaps = []
    for row in g:
        xs = np.flatnonzero(row)
        if len(xs) < 4:
            continue
        split = np.flatnonzero(np.diff(xs) > 8)
        if len(split) != 1:
            continue
        gaps.append(xs[split[0] + 1:].mean() - xs[:split[0] + 1].mean())
    if len(gaps) < 0.6 * g.shape[0]:
        return np.nan, np.nan
    gaps = np.array(gaps) * res
    return gaps.std() / gaps.mean(), gaps.mean()


def estimate_pitch(masks, K, candidates=np.arange(-12.0, 2.01, 0.25)):
    """여러 정지 프레임 마스크로 pitch 추정. (best pitch, candidates, costs (F,P), gaps (F,P)).

    차선이 직선인 구간에서만 성립하는 조건이라 커브 주행 프레임은 넣지 말 것.
    """
    costs = np.full((len(masks), len(candidates)), np.nan)
    gaps = np.full_like(costs, np.nan)
    for a, m in enumerate(masks):
        for b, p in enumerate(candidates):
            costs[a, b], gaps[a, b] = lane_gap_stats(m, K, p)
    valid = ~np.all(np.isnan(costs), axis=0)
    if not valid.any():
        raise ValueError("no pitch candidate sees two lanes; check the lane mask")
    mean_cost = np.where(valid, np.nanmean(np.where(valid, costs, 0.0), axis=0), np.nan)
    best = int(np.nanargmin(mean_cost))
    return float(candidates[best]), np.asarray(candidates), costs, gaps


# ---- waypoint 라벨 --------------------------------------------------------------------------

LABEL_GRID = dict(x_range=(0.3, 2.5), y_range=(-1.5, 1.5), res=0.01)   # 차선 검출용. 2.5 m 넘으면 IPM 이 뭉개짐


def polyfit_xy(P: np.ndarray) -> np.ndarray:
    """y = f(x). 전방 길이가 짧으면 1차, 아니면 2차."""
    return np.polyfit(P[:, 0], P[:, 1], 2 if np.ptp(P[:, 0]) > 0.5 else 1)


def group_lanes(mask_grid, x_range, y_range, res, min_area=40, tol_m=0.08, min_pts=150, min_len_m=0.4):
    """마스크 조각을 차선(최대 2개)으로 묶음. 각 차선은 지면 점 (N,2).

    가림·반사로 한 차선이 여러 조각으로 끊기므로, 큰 조각부터 보면서 기존 차선의 y=f(x) 에
    잔차 중앙값 tol_m 이하로 맞으면 합치고 아니면 새 차선을 시작함.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_grid)
    comps = sorted([k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= min_area],
                   key=lambda k: -stats[k, cv2.CC_STAT_AREA])
    lanes = []
    for k in comps:
        P = grid_to_ground(*np.nonzero(lab == k), x_range, y_range, res)
        best, best_res = None, tol_m
        for j, L in enumerate(lanes):
            r = np.median(np.abs(np.polyval(polyfit_xy(L), P[:, 0]) - P[:, 1]))
            if r < best_res:
                best, best_res = j, r
        if best is not None:
            lanes[best] = np.vstack([lanes[best], P])
        elif len(lanes) < 2:
            lanes.append(P)
    return [L for L in lanes if len(L) >= min_pts and np.ptp(L[:, 0]) >= min_len_m]


def sample_lane(L: np.ndarray, step: float = 0.02) -> np.ndarray:
    xs = np.arange(L[:, 0].min(), L[:, 0].max(), step)
    return np.column_stack([xs, np.polyval(polyfit_xy(L), xs)])


def centerline_points(lanes, lane_width_m: float):
    """중심선 위의 점들과 상태.

    두 차선: 한 차선의 각 점에서 다른 차선의 최근접점을 찾아 중점을 모음 (양방향). 같은 x 에서
             평균 내면 커브에서 두 차선이 x 로 어긋나 있어 틀림. 거리가 폭의 0.4~1.6 배인 쌍만 씀
             (주행 중 pitch 가 조금씩 변해서 펴진 폭도 변함).
    한 차선: 법선 방향으로 폭의 절반만큼 평행이동.
    """
    if len(lanes) == 2:
        A, B = sample_lane(lanes[0]), sample_lane(lanes[1])
        dist = np.linalg.norm(A[:, None] - B[None], axis=2)
        mids = []
        for P, Q, axis in ((A, B, 1), (B, A, 0)):
            j, dmin = dist.argmin(axis=axis), dist.min(axis=axis)
            ok = (dmin > 0.4 * lane_width_m) & (dmin < 1.6 * lane_width_m)
            mids.append((P[ok] + Q[j[ok]]) / 2)
        mids = np.vstack(mids)
        return (mids, "two_lanes") if len(mids) >= 20 else (None, "bad_width")
    if len(lanes) == 1:
        S = sample_lane(lanes[0])
        t = np.gradient(S, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        left_normal = np.column_stack([-t[:, 1], t[:, 0]])
        is_left = S[np.argmin(S[:, 0]), 1] > 0             # 가장 가까운 점이 왼쪽(+y)이면 왼쪽 차선
        mids = S - left_normal * lane_width_m / 2 if is_left else S + left_normal * lane_width_m / 2
        return mids, "one_lane_left" if is_left else "one_lane_right"
    return None, "no_lane"


def waypoint_along(mids: np.ndarray, ahead_m: float, origin=(0.0, 0.0), extrap_m: float = 0.15):
    """gt.waypoint_ahead 와 같은 정의. 중심선에서 origin(후륜축)에 가장 가까운 점부터 호길이 ahead_m.

    중심선은 y=f(x) 로 매끄럽게 한 뒤 origin 쪽으로 늘려서 최근접점을 찾음 (코앞은 카메라 사각지대라
    관측이 없음). 앞쪽은 관측 끝에서 extrap_m 까지만 늘리고, 거기까지 못 가면 nan.
    반환 (wp (2,), 중심선 polyline).
    """
    co = polyfit_xy(mids)
    x0 = min(origin[0], mids[:, 0].min()) - 0.2
    xs = np.arange(x0, mids[:, 0].max() + extrap_m, 0.005)
    C = np.column_stack([xs, np.polyval(co, xs)])
    i0 = int(np.argmin(np.linalg.norm(C - np.asarray(origin), axis=1)))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C[i0:], axis=0), axis=1))])
    if s[-1] < ahead_m:
        return np.full(2, np.nan), C
    j = int(np.searchsorted(s, ahead_m))
    a = (ahead_m - s[j - 1]) / (s[j] - s[j - 1])
    return C[i0 + j - 1] + a * (C[i0 + j] - C[i0 + j - 1]), C


def label_frame(img_undist, H_g2i, lane_width_m: float, ahead_m: float, v_min: float,
                origin=(0.0, 0.0), grid=LABEL_GRID) -> dict:
    """보정된 영상 한 장 -> waypoint 라벨. H_g2i 는 이 영상 해상도의 것 (원본이든 camsim 해상도든).

    마스크는 원본 해상도에서 만들고 나서 폄. 먼저 펴고 색을 고르면 먼 테이프가 보간으로 흐려져 놓침.
    """
    m = warp_to_grid(lane_mask(img_undist, v_min), H_g2i, interp=cv2.INTER_NEAREST, **grid)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lanes = group_lanes(m, **grid)
    mids, status = centerline_points(lanes, lane_width_m)
    if mids is None:
        return dict(wp=np.full(2, np.nan), status=status, lanes=lanes, center=None)
    wp, C = waypoint_along(mids, ahead_m, origin)
    return dict(wp=wp, status=status, lanes=lanes, center=C)


def flag_jumps(wps: np.ndarray, tol_m: float = 0.08, win: int = 3) -> np.ndarray:
    """앞뒤 win 프레임 waypoint 중앙값에서 tol_m 넘게 벗어난 프레임. 검출이 순간적으로 튄 것."""
    wps = np.asarray(wps, dtype=np.float64)
    out = np.zeros(len(wps), bool)
    for n in range(len(wps)):
        nb = np.delete(wps[max(0, n - win):n + win + 1], min(n, win), axis=0)
        nb = nb[np.isfinite(nb[:, 0])]
        if len(nb) and np.isfinite(wps[n, 0]):
            out[n] = np.linalg.norm(wps[n] - np.median(nb, axis=0)) > tol_m
    return out


# ---- 저장 ------------------------------------------------------------------------------

class LabelWriter:
    """dataset.DiskDataset 이 읽는 포맷으로 저장. out_dir/images/NNNNNN.png + labels.csv.

    실차엔 world pose 가 없어서 x, y, theta 는 nan. 학습엔 wp 만 쓰므로 상관없음.
    """

    def __init__(self, out_dir):
        self.out_dir = str(out_dir)
        os.makedirs(os.path.join(self.out_dir, "images"), exist_ok=True)
        self._f = open(os.path.join(self.out_dir, "labels.csv"), "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(LABEL_HEADER)
        self.count = 0

    def add(self, bev_bgr: np.ndarray, wp, name: str = None) -> str:
        name = name or f"{self.count:06d}.png"
        cv2.imwrite(os.path.join(self.out_dir, "images", name), bev_bgr)
        self._w.writerow([name, "nan", "nan", "nan", *np.round(np.asarray(wp, float), 4)])
        self.count += 1
        return name

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
