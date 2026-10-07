"""지면 <-> 이미지 homography.

카메라 가정값(화각/높이/pitch)으로 만들거나, 캘리브레이션으로 실측한 H 파일을 읽음.

좌표계
  vehicle : 후륜축이 원점, x 전방, y 좌측, z 위
  camera  : x 우, y 아래, z 전방 (OpenCV 규약)
  image   : u 우, v 아래 (px)

H_g2i 는 지면 (x, y, 1) -> 이미지 (u, v, w), H_i2g 는 그 역행렬.
"""
import os
import numpy as np
from .config import Config

# pitch 0 일 때 vehicle -> camera 축 대응: x_c = -y_v, y_c = -z_v, z_c = x_v
_R_VC = np.array([[0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [1.0, 0.0, 0.0]])

# build() 가 프레임마다 불릴 수 있어서 실측 H 파일은 경로별로 캐시
_H_FILE_CACHE: dict = {}


def focal_px(cfg: Config) -> float:
    return (cfg.camera.image_width / 2.0) / np.tan(np.deg2rad(cfg.camera.hfov_deg) / 2.0)


def intrinsics(cfg: Config) -> np.ndarray:
    f = focal_px(cfg)
    cx, cy = cfg.camera.image_width / 2.0, cfg.camera.image_height / 2.0
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def extrinsics(cfg: Config, pitch_deg: float) -> np.ndarray:
    """p_c = R p_v + t 의 [R | t] (3x4)."""
    th = np.deg2rad(pitch_deg)
    # 카메라 x 축 회전. pitch 가 +면 광축이 +y_c(아래)로 기움
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(th), -np.sin(th)],
                   [0.0, np.sin(th), np.cos(th)]])
    R = Rx @ _R_VC
    C = np.array([cfg.camera.offset_x_m, 0.0, cfg.camera.height_m])   # vehicle 기준 카메라 중심
    t = -R @ C
    return np.hstack([R, t[:, None]])


def _load_h_i2g_file(path: str) -> np.ndarray:
    key = os.path.abspath(path)
    H = _H_FILE_CACHE.get(key)
    if H is None:
        H = np.load(key).astype(np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"h_i2g_file must be 3x3, got {H.shape}")
        _H_FILE_CACHE[key] = H
    return H


def _assumed_h_g2i(cfg: Config, pitch_deg: float) -> np.ndarray:
    """실측 파일 말고 카메라 가정값으로만 만든 H_g2i."""
    Rt = extrinsics(cfg, pitch_deg)
    H_g2i = intrinsics(cfg) @ Rt[:, [0, 1, 3]]            # 지면은 z_v = 0 이라 3번째 열 뺌
    # H_g2i[2, 2] 로 정규화하면 안 됨. 기본 config(offset_x_m=0, pitch_deg=0)에서는 그 값이
    # 정확히 0이라 NaN 됨 (지면 원점이 카메라 바로 아래라 전방 거리가 0).
    # project() 는 스케일에 무관하므로 Frobenius norm 으로 나눠도 결과 똑같음.
    H_g2i /= np.linalg.norm(H_g2i)
    return H_g2i


def build(cfg: Config, pitch_deg=None):
    """(H_g2i, H_i2g). h_i2g_file 이 설정돼 있으면 그쪽이 카메라 가정값을 이김.

    실측 H 쓰면서 pitch 를 흔들고 싶을 때(증강)가 문제인데, 실측 H 에는 pitch 정보가 이미 녹아 있어서
    그냥 갈아끼울 수가 없음. 그래서 가정 카메라의 pitch 차이만큼만 보정해 곱함:
    H_g2i = H_file @ inv(H_assumed(설정 pitch)) @ H_assumed(요청 pitch).
    """
    if cfg.camera.h_i2g_file:
        H_i2g_file = _load_h_i2g_file(cfg.camera.h_i2g_file)
        H_g2i_file = np.linalg.inv(H_i2g_file)
        if pitch_deg is not None and pitch_deg != cfg.camera.pitch_deg:
            H_assumed_base = _assumed_h_g2i(cfg, cfg.camera.pitch_deg)
            H_assumed_new = _assumed_h_g2i(cfg, pitch_deg)
            H_g2i = H_g2i_file @ np.linalg.inv(H_assumed_base) @ H_assumed_new
            H_g2i /= np.linalg.norm(H_g2i)
            return H_g2i, np.linalg.inv(H_g2i)
        return H_g2i_file, H_i2g_file
    if pitch_deg is None:
        pitch_deg = cfg.camera.pitch_deg
    H_g2i = _assumed_h_g2i(cfg, pitch_deg)
    return H_g2i, np.linalg.inv(H_g2i)


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    hom = np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1) @ H.T
    return hom[..., :2] / hom[..., 2:3]
