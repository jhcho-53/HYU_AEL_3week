"""테이프 트랙을 그림. 앞에서 본 카메라 뷰와 위에서 본 BEV 두 가지.

모델에 넣는 건 BEV 쪽. 원근 렌더(render)는 눈으로 확인하거나 IPM 과 비교할 때 씀.
"""
import cv2
import numpy as np
from .config import Config
from .camera import project

# fillPoly 의 고정소수점 좌표. 1/16 px 단위로 그려서 테이프 가장자리가 계단지지 않게
_SHIFT = 4
_SCALE = 1 << _SHIFT


def to_vehicle(pose, pts_world: np.ndarray) -> np.ndarray:
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    A = np.array([[c, -s], [s, c]])          # A 의 행이 R(th) 라서 (p - o) @ A 가 곧 R(-th)(p - o)
    return (np.asarray(pts_world, dtype=np.float64) - [x, y]) @ A


def visible_quads(qv: np.ndarray, scan, cfg: Config) -> np.ndarray:
    """그릴 quad 만 골라내는 bool 마스크. qv 는 (M,4,2) 차량 좌표계.

    near/far 컷은 차량 원점이 아니라 **카메라** 기준이어야 함. offset_x_m > 0 이면 카메라가
    후륜축보다 앞에 있어서, 차량 기준 x 가 0~offset_x_m 인 quad 는 실제로는 카메라 뒤에 있음.
    이걸 놓치면 깊이가 음수인 채로 투영돼서 지평선 위에 거꾸로 나타남. 실제로 겪은 버그.

    LiDAR 가림은 반대로 차량 기준. scan 이 카메라가 아니라 차량 pose 에서 나오기 때문.
    """
    ctr = qv.mean(1)
    off = cfg.camera.offset_x_m
    xc = qv[:, :, 0] - off                             # 카메라 기준 전방 거리
    rng = np.hypot(ctr[:, 0], ctr[:, 1])               # 차량 원점 기준. LiDAR 용
    cam_rng = np.hypot(ctr[:, 0] - off, ctr[:, 1])     # 카메라 기준. far 컷 용
    keep = (xc.min(1) > cfg.render.near_m) & (cam_rng < cfg.render.far_m)
    if scan is not None:
        scan = np.asarray(scan)
        fov = cfg.render.lidar_fov_rad
        brg = np.arctan2(ctr[:, 1], ctr[:, 0])
        # gym 은 빔 i 를 -fov/2 + i*fov/(n-1) 에 둠. n-1 스텝이 fov 를 덮는다는 뜻
        idx = np.rint((brg + fov / 2.0) / fov * (len(scan) - 1)).astype(int).clip(0, len(scan) - 1)
        keep &= rng < scan[idx]
    return keep


def render(pose, quads_world: np.ndarray, scan, H_g2i: np.ndarray, cfg: Config) -> np.ndarray:
    """앞에서 본 카메라 뷰. scan 을 주면 다른 차에 가린 테이프는 빠짐."""
    W, Hh = cfg.camera.image_width, cfg.camera.image_height
    img = np.empty((Hh, W, 3), np.uint8)
    img[:] = cfg.lane.color_floor
    qv = to_vehicle(pose, quads_world)
    keep = visible_quads(qv, scan, cfg)
    if keep.any():
        uv = project(H_g2i, qv[keep])                       # (K,4,2)
        polys = np.round(uv * _SCALE).astype(np.int32)
        cv2.fillPoly(img, list(polys), tuple(int(c) for c in cfg.lane.color_tape),
                     lineType=cv2.LINE_AA, shift=_SHIFT)
    return img


def draw_points(img, pts_vehicle, H_g2i, color=(0, 255, 0), radius=4):
    uv = project(H_g2i, np.asarray(pts_vehicle, float).reshape(-1, 2))
    for u, v in uv:
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (int(round(u)), int(round(v))), radius, color, -1)
    return img


# ---- BEV (위에서 본 그림) -----------------------------------------------------
# 픽셀 규약: 위 = 전방(+x), 왼쪽 = 차량 좌측(+y). 범위와 해상도는 config 의 bev 섹션

def bev_size(cfg: Config):
    b = cfg.bev
    return (int(round((b.x_range_m[1] - b.x_range_m[0]) / b.resolution_m)),
            int(round((b.y_range_m[1] - b.y_range_m[0]) / b.resolution_m)))


def bev_pixels(pts_vehicle: np.ndarray, cfg: Config) -> np.ndarray:
    """차량 좌표계 지면점 (...,2) m -> BEV 픽셀 (...,2)."""
    b = cfg.bev
    p = np.asarray(pts_vehicle, dtype=np.float64)
    u = (b.y_range_m[1] - p[..., 1]) / b.resolution_m
    v = (b.x_range_m[1] - p[..., 0]) / b.resolution_m
    return np.stack([u, v], axis=-1)


def ground_to_bev_matrix(cfg: Config) -> np.ndarray:
    """bev_pixels 와 같은 변환을 3x3 행렬로. warpPerspective 에 넘길 때 씀."""
    b = cfg.bev
    r = b.resolution_m
    return np.array([[0.0, -1.0 / r, b.y_range_m[1] / r],
                     [-1.0 / r, 0.0, b.x_range_m[1] / r],
                     [0.0, 0.0, 1.0]])


def bev_visibility_mask(H_g2i: np.ndarray, cfg: Config) -> np.ndarray:
    """BEV 픽셀 중 카메라가 실제로 볼 수 있는 곳만 True 인 (h,w) 마스크.

    실차 IPM 출력은 화각 밖과 코앞 사각지대가 비어 있음. 시뮬 BEV 도 같은 데를 가려줘야
    모델이 보는 그림이 실차와 같아짐. 픽셀 중심을 지면 좌표로 바꿔 카메라에 투영해 보고,
    이미지 안에 떨어지는지(깊이 > 0) near 컷을 통과하는지 봄.
    """
    h, w = bev_size(cfg)
    b = cfg.bev
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x = b.x_range_m[1] - vs * b.resolution_m
    y = b.y_range_m[1] - us * b.resolution_m
    hom = np.stack([x, y, np.ones_like(x)], -1) @ H_g2i.T
    depth_ok = hom[..., 2] > 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        u = hom[..., 0] / hom[..., 2]
        v = hom[..., 1] / hom[..., 2]
    in_img = (u >= 0) & (u < cfg.camera.image_width) & (v >= 0) & (v < cfg.camera.image_height)
    ahead = (x - cfg.camera.offset_x_m) > cfg.render.near_m
    return depth_ok & in_img & ahead


def render_bev(pose, quads_world: np.ndarray, cfg: Config, mask: np.ndarray = None) -> np.ndarray:
    """모델 입력용 BEV. 테이프를 지오메트리에서 바로 위에서 내려다본 모양으로 그림.

    원근 렌더나 IPM 을 거치지 않아 빠르고 정확함. mask(bev_visibility_mask)를 주면
    카메라가 못 보는 영역이 바닥색으로 덮여 실차 IPM 출력과 같은 모양이 됨.
    """
    h, w = bev_size(cfg)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = cfg.lane.color_floor
    qv = to_vehicle(pose, quads_world)
    b = cfg.bev
    ctr = qv.mean(1)
    keep = ((ctr[:, 0] > b.x_range_m[0] - 0.5) & (ctr[:, 0] < b.x_range_m[1] + 0.5) &
            (ctr[:, 1] > b.y_range_m[0] - 0.5) & (ctr[:, 1] < b.y_range_m[1] + 0.5))
    if keep.any():
        polys = np.round(bev_pixels(qv[keep], cfg) * _SCALE).astype(np.int32)
        cv2.fillPoly(img, list(polys), tuple(int(c) for c in cfg.lane.color_tape),
                     lineType=cv2.LINE_AA, shift=_SHIFT)
    if mask is not None:
        img[~mask] = cfg.lane.color_floor
    return img


def draw_points_bev(img_bev, pts_vehicle, cfg: Config, color=(0, 255, 0), radius=5):
    for u, v in bev_pixels(np.asarray(pts_vehicle, float).reshape(-1, 2), cfg):
        if 0 <= u < img_bev.shape[1] and 0 <= v < img_bev.shape[0]:
            cv2.circle(img_bev, (int(round(u)), int(round(v))), radius, color, -1, cv2.LINE_AA)
    return img_bev


def ipm_bev(img_perspective: np.ndarray, H_i2g: np.ndarray, cfg: Config) -> np.ndarray:
    """실차가 쓰는 경로. 원근 영상을 H_i2g 로 지면에 펴서 BEV 규격으로 맞춤."""
    h, w = bev_size(cfg)
    H_img2bev = ground_to_bev_matrix(cfg) @ H_i2g
    floor = tuple(int(c) for c in cfg.lane.color_floor)
    return cv2.warpPerspective(img_perspective, H_img2bev, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=floor)
