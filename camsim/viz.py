"""지도와 주행 경로 그리기. 학습엔 안 쓰고 눈으로 확인하는 용도.

좌표계가 셋이라 헷갈리기 쉬움.
  world  : gym 맵 좌표 (m)
  map px : examples/*.png 픽셀. col = (x - ox)/res, row = H-1 - (y - oy)/res  (gym 이 상하 반전해 씀)
  canvas : 확대 뷰 픽셀. 차를 가운데 두고 **위가 차량 전방**이 되게 회전한 좌표
"""
import os
import cv2
import numpy as np
import yaml
from .config import Config
from .track import Track
from .render import to_vehicle, bev_pixels

CAR_LEN_M, CAR_WID_M = 0.58, 0.31          # gym 기본 차체. 표시용이라 config 와 별개
COL_WP, COL_CAR, COL_FOV, COL_CENTER = (0, 255, 0), (255, 80, 0), (255, 200, 0), (90, 90, 90)


def to_world(pose, pts_vehicle: np.ndarray) -> np.ndarray:
    """render.to_vehicle 의 역변환."""
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return np.asarray(pts_vehicle, dtype=np.float64) @ R.T + [x, y]


class MapImage:
    """맵 PNG + yaml(origin, resolution). world <-> 픽셀 변환만 담당."""

    def __init__(self, yaml_path: str):
        meta = yaml.safe_load(open(yaml_path))
        img_path = os.path.join(os.path.dirname(yaml_path), meta["image"])
        self.gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.gray is None:
            raise FileNotFoundError(img_path)
        self.res = float(meta["resolution"])
        self.ox, self.oy = float(meta["origin"][0]), float(meta["origin"][1])
        self.h, self.w = self.gray.shape

    def world_to_px(self, pts_world: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_world, dtype=np.float64)
        col = (p[..., 0] - self.ox) / self.res
        row = (self.h - 1) - (p[..., 1] - self.oy) / self.res
        return np.stack([col, row], axis=-1)


def _poly(pts, shift=4):
    return np.round(np.asarray(pts) * (1 << shift)).astype(np.int32)


def draw_track_on_map(mapimg: MapImage, track: Track, cfg: Config, crop_margin_m: float = 2.0):
    """맵에 테이프 그리고 트랙 주변만 잘라서 (img_bgr, offset_px) 로 반환.

    offset 은 잘라낸 좌상단의 원본 픽셀 좌표. 이후 이 그림 위에 뭘 더 그리려면 다 빼줘야 함.
    """
    img = cv2.cvtColor(mapimg.gray, cv2.COLOR_GRAY2BGR)
    tape = tuple(int(c) for c in cfg.lane.color_tape)
    px = mapimg.world_to_px(track.quads)                     # (M,4,2)
    cv2.fillPoly(img, list(_poly(px)), tape, lineType=cv2.LINE_AA, shift=4)
    lo = np.floor(px.reshape(-1, 2).min(0) - crop_margin_m / mapimg.res).astype(int).clip(0)
    hi = np.ceil(px.reshape(-1, 2).max(0) + crop_margin_m / mapimg.res).astype(int)
    hi = np.minimum(hi, [mapimg.w, mapimg.h])
    return img[lo[1]:hi[1], lo[0]:hi[0]].copy(), lo


def mark_poses_on_map(map_bgr: np.ndarray, offset_px, mapimg: MapImage, poses, labels=None, radius=10):
    """잘라낸 맵 위에 pose 를 번호 붙은 원과 헤딩 선으로 찍음."""
    for k, pose in enumerate(poses):
        c = mapimg.world_to_px(pose[:2]) - offset_px
        tip = mapimg.world_to_px(to_world(pose, [[3.0, 0.0]])[0]) - offset_px
        c, tip = tuple(np.round(c).astype(int)), tuple(np.round(tip).astype(int))
        cv2.line(map_bgr, c, tip, COL_CAR, 2, cv2.LINE_AA)
        cv2.circle(map_bgr, c, radius, COL_CAR, -1, cv2.LINE_AA)
        cv2.putText(map_bgr, str(labels[k] if labels else k), (c[0] + radius + 2, c[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return map_bgr


def local_view(pose, track: Track, wp_vehicle, cfg: Config, mapimg: MapImage = None,
               ahead_m: float = 5.0, behind_m: float = 1.5, half_width_m: float = 3.0,
               res_m: float = 0.01) -> np.ndarray:
    """차 주변 확대 top-down. 위가 차량 전방. 벽·테이프·차체·카메라 화각·BEV 범위·GT waypoint(점 하나)를 얹음."""
    h, w = int(round((ahead_m + behind_m) / res_m)), int(round(2 * half_width_m / res_m))

    def vp(pts_v):   # vehicle m -> canvas px
        p = np.asarray(pts_v, dtype=np.float64)
        return np.stack([(half_width_m - p[..., 1]) / res_m, (ahead_m - p[..., 0]) / res_m], axis=-1)

    img = np.full((h, w, 3), 235, np.uint8)
    if mapimg is not None:
        # 맵의 벽을 캔버스로 옮김. 세 점의 대응만 알면 affine 이 정해짐
        src = mapimg.world_to_px(to_world(pose, [[0, 0], [1, 0], [0, 1]])).astype(np.float32)
        dst = vp([[0, 0], [1, 0], [0, 1]]).astype(np.float32)
        A = cv2.getAffineTransform(src, dst)
        occ = cv2.warpAffine(mapimg.gray, A, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
        img[occ < 128] = (40, 40, 40)

    qv = to_vehicle(pose, track.quads)
    ctr = qv.mean(1)
    keep = (ctr[:, 0] > -behind_m - 1) & (ctr[:, 0] < ahead_m + 1) & (np.abs(ctr[:, 1]) < half_width_m + 1)
    tape = tuple(int(c) for c in cfg.lane.color_tape)
    cv2.fillPoly(img, list(_poly(vp(qv[keep]))), tape, lineType=cv2.LINE_AA, shift=4)

    cv = to_vehicle(pose, track.center)
    kc = (cv[:, 0] > -behind_m - 1) & (cv[:, 0] < ahead_m + 1) & (np.abs(cv[:, 1]) < half_width_m + 1)
    for p in vp(cv[kc]):
        cv2.circle(img, tuple(np.round(p).astype(int)), 1, COL_CENTER, -1)

    b = cfg.bev                                            # BEV 범위는 점선 사각형으로
    rect = vp([[b.x_range_m[0], b.y_range_m[1]], [b.x_range_m[1], b.y_range_m[1]],
               [b.x_range_m[1], b.y_range_m[0]], [b.x_range_m[0], b.y_range_m[0]]])
    cv2.polylines(img, [_poly(rect)], True, (200, 120, 200), 1, cv2.LINE_AA, shift=4)

    half = np.deg2rad(cfg.camera.hfov_deg) / 2             # 카메라 화각은 반투명 부채꼴로
    ox = cfg.camera.offset_x_m
    far = cfg.render.far_m
    wedge = vp([[ox, 0], [ox + far * np.cos(half), far * np.sin(half)], [ox + far * np.cos(half), -far * np.sin(half)]])
    overlay = img.copy()
    cv2.fillPoly(overlay, [_poly(wedge)], COL_FOV, lineType=cv2.LINE_AA, shift=4)
    img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)

    # 차체. 원점이 후륜축이라 사각형을 앞으로 뻗어 그림
    car = vp([[-0.1, CAR_WID_M / 2], [CAR_LEN_M - 0.1, CAR_WID_M / 2],
              [CAR_LEN_M - 0.1, -CAR_WID_M / 2], [-0.1, -CAR_WID_M / 2]])
    cv2.fillPoly(img, [_poly(car)], COL_CAR, lineType=cv2.LINE_AA, shift=4)
    a0, a1 = vp([[0, 0], [1.0, 0]])
    cv2.arrowedLine(img, tuple(np.round(a0).astype(int)), tuple(np.round(a1).astype(int)), COL_CAR, 2, cv2.LINE_AA, tipLength=0.25)

    c = tuple(np.round(vp(np.asarray(wp_vehicle, float).reshape(2))).astype(int))
    cv2.circle(img, c, 6, COL_WP, -1, cv2.LINE_AA)
    cv2.putText(img, f"{cfg.waypoints.ahead_m:g} m", (c[0] + 8, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 90, 0), 1, cv2.LINE_AA)
    return img


def side_by_side(*imgs, gap=10, bg=255) -> np.ndarray:
    """높이 맞춰 가로로 이어 붙임."""
    hmax = max(i.shape[0] for i in imgs)
    out = []
    for i in imgs:
        s = hmax / i.shape[0]
        r = cv2.resize(i, (int(round(i.shape[1] * s)), hmax), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        out += [r, np.full((hmax, gap, 3), bg, np.uint8)]
    return np.hstack(out[:-1])


PATH_COLORS = [(110, 110, 110), (255, 0, 0), (0, 160, 0), (0, 0, 255), (200, 0, 200), (0, 140, 255), (120, 120, 0)]  # 0번은 회색. 오라클/기준 경로용


def magnify_offsets(track: Track, xy: np.ndarray, factor: float) -> np.ndarray:
    """중심선 대비 편차만 factor 배로 부풀림. 경로가 놓인 위치 자체는 유지.

    160 m 트랙에서 주행 경로들의 차이는 수 cm 라 전체 맵에 그리면 1~2 px 로 겹쳐 버림.
    비교해 보려고 그림에서만 과장하는 것이고, 실제 위치 아님. factor=1 이면 원본.
    """
    xy = np.asarray(xy, float)[:, :2]
    if factor == 1.0 or len(xy) == 0:
        return xy
    d = ((xy[:, None, :] - track.center[None, :, :]) ** 2).sum(-1)
    i = np.argmin(d, axis=1)
    base = track.center[i]
    return base + factor * (xy - base)


def draw_paths_on_map(mapimg: MapImage, track: Track, cfg: Config, paths: dict, thickness: int = 1,
                      crop_margin_m: float = 2.0, legend: bool = True, dashed: bool = True,
                      magnify: float = 1.0):
    """맵 위에 GT 중심선(검은 점선)과 주행 경로들을 겹쳐 그림.

    경로들이 거의 겹쳐서 얇게 그리고, dashed=True 면 파선 패턴과 위상을 달리해 구분.
    paths 는 {라벨: (N,2) world xy 또는 (N,3) pose}. 반환은 (img, offset_px) 이고
    offset 에는 범례 띠 높이가 이미 반영돼 있음.
    """
    img, off = draw_track_on_map(mapimg, track, cfg, crop_margin_m)
    c = np.round(mapimg.world_to_px(track.center) - off).astype(np.int32)
    for k in range(0, len(c), 6):                      # 중심선은 점선
        cv2.line(img, tuple(c[k]), tuple(c[(k + 3) % len(c)]), (0, 0, 0), 1, cv2.LINE_AA)

    def _draw(dst, px, col, on, off_, phase):
        """(on, off_) 픽셀 패턴의 파선. on=0 이면 실선."""
        if on <= 0:
            cv2.polylines(dst, [px.reshape(-1, 1, 2)], False, col, thickness, cv2.LINE_AA)
            return
        seg = np.hypot(*np.diff(px, axis=0).T)
        pos = np.concatenate([[0.0], np.cumsum(seg)])
        keep = ((pos + phase) % (on + off_)) < on
        for k in range(len(px) - 1):
            if keep[k]:
                cv2.line(dst, tuple(px[k]), tuple(px[k + 1]), col, thickness, cv2.LINE_AA)

    band = (20 * (len(paths) + (1 if magnify != 1.0 else 0)) + 12) if legend else 0
    if band:                                             # 범례용 흰 띠를 그림 위에 덧댐
        img = np.vstack([np.full((band, img.shape[1], 3), 255, np.uint8), img])
        off = np.array([off[0], off[1] - band])
    y0 = 20
    if legend and magnify != 1.0:
        cv2.putText(img, f"lateral deviation x{magnify:g} (comparison only)", (12, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y0 += 20
    for i, (label, xy) in enumerate(paths.items()):
        col = PATH_COLORS[i % len(PATH_COLORS)]
        on, off_ = (0, 0) if not dashed or i == 0 else (14, 8)
        phase = 0 if not dashed else i * 6              # 패턴이 같아도 위상을 어긋나게
        xy = magnify_offsets(track, xy, magnify)
        if len(xy) == 0:
            continue
        px = np.round(mapimg.world_to_px(xy) - off).astype(np.int32)
        _draw(img, px, col, on, off_, phase)
        cv2.circle(img, tuple(px[0]), 5, col, 1, cv2.LINE_AA)          # 시작은 빈 원
        cv2.circle(img, tuple(px[-1]), 6, col, -1, cv2.LINE_AA)        # 끝은 채운 원
        if legend:
            sample = np.array([[12, y0 - 5], [56, y0 - 5]], np.int32)  # 범례에도 같은 패턴으로
            _draw(img, sample, col, on, off_, phase)
            cv2.putText(img, label, (64, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            y0 += 20
    return img, off


def crop_around(img: np.ndarray, offset_px, mapimg: MapImage, xy_world, half_m: float = 4.0, scale: int = 3):
    """맵 그림에서 world 한 점 주변을 잘라 확대."""
    c = np.round(mapimg.world_to_px(np.asarray(xy_world, float)[:2]) - offset_px).astype(int)
    r = int(half_m / mapimg.res)
    x0, y0 = max(c[0] - r, 0), max(c[1] - r, 0)
    sub = img[y0:c[1] + r, x0:c[0] + r]
    return cv2.resize(sub, (sub.shape[1] * scale, sub.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def to_h264(src: str, dst: str = None, max_width: int = 960, crf: int = 28) -> str:
    """OpenCV 가 쓴 mp4v 를 H.264 로 다시 인코딩.

    브라우저와 VSCode 의 HTML5 비디오는 mp4v 를 재생 못 해서, 이걸 안 거치면 셀에 검은 화면만 뜸.
    ffmpeg 는 imageio-ffmpeg 가 번들한 걸 쓰므로 시스템에 따로 설치할 필요 없음.
    """
    import subprocess
    import imageio_ffmpeg
    if dst is None:
        base, _ = os.path.splitext(src)
        dst = base + "_h264.mp4"
    vf = f"scale='min({max_width},iw)':-2"          # 폭 제한, 높이는 짝수로 (yuv420p 요구사항)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src,
           "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart", dst]
    subprocess.run(cmd, check=True)
    return dst
