"""BEV 증강. sim-to-real 갭 줄이려고 시뮬 그림을 일부러 지저분하게 만듦.

지켜야 할 규칙 하나. 라벨(waypoint)은 지오메트리 참값이라 증강으로 안 바뀜.
그래서 **정답을 옮기는 변형(이동·회전·스케일)은 넣으면 안 됨.** "같은 장면을 다르게 본 것"만 됨.
예외는 jitter_bev 하나. 실차에서 pitch 변하면 IPM 결과 자체가 휘니까 그건 진짜 관측 변화.

전부 f(bev, cfg, rng) -> bev 꼴이고, 세기는 config.yaml 의 augment 섹션에서 옴.
기본값은 전부 "변화 없음"이라 학생이 값을 켜야 효과 남.

주의: jitter_bev 말고는 실제 카메라 물리의 대충 근사. 실차 영상 찍어 보고 다시 설계하는 게 맞음.
"""
import cv2
import numpy as np
from .config import Config
from .camera import build
from .render import ground_to_bev_matrix


# ---- 기하 -------------------------------------------------------------------

def jitter_bev(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """pitch 가 틀어진 카메라 영상을 공칭 H_i2g 로 IPM 했을 때 생기는 왜곡.

    가감속하면 서스펜션이 물러서 카메라가 까딱하고, IPM 의 평면 가정이 깨짐. 먼 곳일수록 크게 휨.
    경로는 참 지면점 -> (pitch+δ 카메라) 이미지 -> (공칭 H_i2g) 지면 -> BEV 픽셀.
    """
    j = cfg.augment.pitch_jitter_deg
    if j <= 0:
        return bev
    d = rng.uniform(-j, j)
    H_g2i_true = build(cfg, pitch_deg=cfg.camera.pitch_deg + d)[0]
    H_i2g_nom = build(cfg)[1]
    A = ground_to_bev_matrix(cfg)
    M = A @ H_i2g_nom @ H_g2i_true @ np.linalg.inv(A)
    h, w = bev.shape[:2]
    return cv2.warpPerspective(bev, M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=tuple(int(c) for c in cfg.lane.color_floor))


def ipm_blur(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """전방 거리에 비례해 커지는 블러.

    실차 IPM 은 먼 곳일수록 카메라 픽셀 하나가 넓은 바닥을 덮어서 뭉개짐. 거리별로 정확히
    계산하는 대신 행 구간을 나눠 점점 강한 GaussianBlur 를 이어 붙임.
    """
    kmax = int(cfg.augment.ipm_blur_max_px)
    if kmax < 3:
        return bev
    h = bev.shape[0]
    levels = list(range(3, kmax + 1, 2))                     # 커널은 홀수만
    if not levels:
        return bev
    out = bev.copy()
    edges = np.linspace(0, h, len(levels) + 1).astype(int)   # 위(먼 곳)부터 강하게
    for k, lvl in enumerate(reversed(levels)):
        y0, y1 = edges[k], edges[k + 1]
        pad = lvl                                            # 구간 경계에 선 안 생기게 여유 두고 블러
        a, b = max(0, y0 - pad), min(h, y1 + pad)
        blurred = cv2.GaussianBlur(bev[a:b], (lvl, lvl), 0)
        out[y0:y1] = blurred[y0 - a:y1 - a]
    return out


def erase_patches(bev: np.ndarray, cfg: Config, rng: np.random.Generator, n_max: int = 3,
                  size_m=(0.05, 0.30)) -> np.ndarray:
    """임의 사각형을 바닥색으로 지움. 테이프가 벗겨졌거나 뭔가에 가린 상황. 크기는 m 라 BEV 해상도와 무관."""
    if rng.uniform() >= cfg.augment.tape_dropout_prob:
        return bev
    out = bev.copy()
    h, w = out.shape[:2]
    floor = np.array(cfg.lane.color_floor, np.uint8)
    lo, hi = (max(1, int(round(s / cfg.bev.resolution_m))) for s in size_m)
    for _ in range(int(rng.integers(1, n_max + 1))):
        ph, pw = rng.integers(lo, hi + 1, 2)
        y, x = int(rng.integers(0, max(1, h - ph))), int(rng.integers(0, max(1, w - pw)))
        out[y:y + ph, x:x + pw] = floor
    return out


# ---- 조명 -------------------------------------------------------------------

def brightness_contrast(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """노출·조도 변화. convertScaleAbs 의 alpha(대비) / beta(밝기)."""
    a = cfg.augment.contrast_range
    b = cfg.augment.brightness_delta
    alpha = rng.uniform(a[0], a[1])
    beta = rng.uniform(-b, b) if b > 0 else 0.0
    if alpha == 1.0 and beta == 0.0:
        return bev
    return cv2.convertScaleAbs(bev, alpha=alpha, beta=beta)


def gamma(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """카메라 톤커브 차이. 어두운 쪽 디테일이 특히 달라짐."""
    lo, hi = cfg.augment.gamma_range
    if lo == 1.0 and hi == 1.0:
        return bev
    g = rng.uniform(lo, hi)
    lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / g)) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(bev, lut)


def hsv_shift(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """조명 색온도. 백열등 아래 노란 테이프와 형광등 아래 노란 테이프는 다른 색."""
    dh = cfg.augment.hue_shift_deg
    slo, shi = cfg.augment.sat_scale
    if dh == 0 and slo == 1.0 and shi == 1.0:
        return bev
    hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV).astype(np.int16)
    if dh:
        hsv[..., 0] = (hsv[..., 0] + int(rng.uniform(-dh, dh) / 2)) % 180   # OpenCV hue 는 0..179
    if not (slo == 1.0 and shi == 1.0):
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(slo, shi), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def illumination(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """천장 조명이 한쪽만 밝은 실습실. 부드러운 밝기 기울기를 곱함.

    세기 s 면 배율이 1-s ~ 1+s 사이에서 화면을 가로지르며 완만히 변함. 방향은 매번 랜덤.
    """
    s = float(cfg.augment.illum_strength)
    if s <= 0:
        return bev
    h, w = bev.shape[:2]
    th = rng.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (np.cos(th) * xs / w + np.sin(th) * ys / h)
    t = (t - t.min()) / max(float(t.max() - t.min()), 1e-6) * 2 - 1        # -1..1 로 정규화
    field = (1.0 + s * t)[..., None]
    return np.clip(bev.astype(np.float32) * field, 0, 255).astype(np.uint8)


def shadow(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """구조물이나 사람 그림자. 볼록 다각형을 어둡게 하고 경계는 블러로 흐림."""
    if rng.uniform() >= cfg.augment.shadow_prob:
        return bev
    h, w = bev.shape[:2]
    n = int(rng.integers(3, 6))
    # 화면 밖까지 꼭짓점을 뿌려야 그림자가 화면을 가로질러 걸친 모양이 나옴
    pts = np.column_stack([rng.integers(-w // 4, w + w // 4, n), rng.integers(-h // 4, h + h // 4, n)])
    mask = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts.astype(np.int32)), 1.0, cv2.LINE_AA)
    k = max(3, (min(h, w) // 20) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
    dark = rng.uniform(cfg.augment.shadow_darkness[0], cfg.augment.shadow_darkness[1])
    return np.clip(bev.astype(np.float32) * (1 - mask * (1 - dark)), 0, 255).astype(np.uint8)


# ---- 센서 -------------------------------------------------------------------

def blur(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """초점 흐림. 글로벌 셔터라 실제 모션 블러는 작아서 이 정도로 뭉뚱그림."""
    kmax = int(cfg.augment.blur_max_px)
    if kmax < 3:
        return bev
    k = int(rng.integers(1, (kmax + 1) // 2 + 1)) * 2 - 1     # 홀수 커널
    return bev if k < 3 else cv2.GaussianBlur(bev, (k, k), 0)


def noise(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """게인 올렸을 때의 센서 노이즈."""
    s = float(cfg.augment.noise_sigma)
    if s <= 0:
        return bev
    return np.clip(bev.astype(np.float32) + rng.normal(0, s, bev.shape), 0, 255).astype(np.uint8)


def jpeg(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """JPEG 아티팩트. ROS image_transport 의 compressed 쓰면 실차 입력에 이게 섞임."""
    lo, hi = cfg.augment.jpeg_quality
    if lo >= 100 and hi >= 100:
        return bev
    q = int(rng.integers(lo, hi + 1))
    ok, enc = cv2.imencode(".jpg", bev, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else bev


# ---- 조합 -------------------------------------------------------------------

def example_augment(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """기하 -> 조명 -> 센서 순으로 다 적용해 본 체인. 실제 촬영에서 일어나는 순서.

    config 세기가 전부 기본값이면 그냥 항등 함수. 노트북 과제에서 my_augment 의 출발점으로 씀.
    """
    for fn in (jitter_bev, ipm_blur, erase_patches,
               illumination, shadow, brightness_contrast, gamma, hsv_shift,
               blur, noise, jpeg):
        bev = fn(bev, cfg, rng)
    return bev
