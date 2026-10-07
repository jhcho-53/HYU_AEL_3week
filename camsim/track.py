"""중심선 CSV 를 트랙 지오메트리로.

등간격 중심선, 각 지점의 헤딩과 누적 호길이, 바닥에 붙일 테이프 사각형 목록을 만듦.
테이프는 선 하나가 아니라 짧은 사각형(quad) 수천 개. 그래야 렌더러가 fillPoly 한 번으로 그림.
"""
from dataclasses import dataclass
import numpy as np
import os

import yaml

from .config import Config


@dataclass
class Track:
    center: np.ndarray    # (N,2) 기준 경로, world m
    heading: np.ndarray   # (N,) rad
    s: np.ndarray         # (N,) 누적 호길이, s[0]=0
    length: float         # 한 바퀴 길이
    quads: np.ndarray     # (M,4,2) 테이프 사각형, world m
    left_m: np.ndarray    # (N,) 기준선에서 왼쪽 테이프까지 거리 (법선 + 방향)
    right_m: np.ndarray   # (N,) 오른쪽 테이프까지 거리 (법선 - 방향)


def resample(xy: np.ndarray, step: float) -> np.ndarray:
    """닫힌 경로를 등간격으로 다시 찍음. 마지막 점은 시작점과 겹치지 않게 뺌."""
    xy = np.asarray(xy, dtype=np.float64)
    if np.hypot(*(xy[0] - xy[-1])) > 1e-6:
        xy = np.vstack([xy, xy[:1]])
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(np.floor(s[-1] / step))
    s_new = np.arange(n) * step
    return np.column_stack([np.interp(s_new, s, xy[:, 0]), np.interp(s_new, s, xy[:, 1])])


def _heading(center: np.ndarray) -> np.ndarray:
    # 앞뒤 점을 잇는 중앙차분. 한쪽 차분보다 코너에서 덜 떨림
    d = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def _tape_quads(center, heading, offset, tape_w):
    """offset 은 스칼라도 되고 지점별 (N,) 배열도 됨 (벽 따라가면 폭이 변함)."""
    nrm = np.column_stack([-np.sin(heading), np.cos(heading)])
    line = center + np.asarray(offset).reshape(-1, 1) * nrm if np.ndim(offset) else center + offset * nrm
    nxt = np.roll(line, -1, axis=0)
    nrm_n = np.roll(nrm, -1, axis=0)
    h = tape_w / 2.0
    return np.stack([line - h * nrm, nxt - h * nrm_n, nxt + h * nrm_n, line + h * nrm], axis=1)


def wall_offsets(center: np.ndarray, heading: np.ndarray, map_yaml: str, margin_m: float,
                 max_search_m: float = 6.0, step_m: float = 0.02):
    """중심선에서 좌우로 벽까지 거리를 재고, 거기서 margin_m 안쪽 위치를 돌려줌.

    맵 PNG 를 흑백으로 읽어 밝은 픽셀을 자유 공간으로 보고, 법선 방향으로 2cm 씩 전진하다
    벽 만나면 멈추는 단순 레이마칭. 반환값 둘 다 양수, 각각 +normal / -normal 거리(m).
    """
    from PIL import Image        # cv2 없이도 트랙 만들 수 있게. Pillow 는 gym 이 이미 의존함
    meta = yaml.safe_load(open(map_yaml))
    img = np.asarray(Image.open(os.path.join(os.path.dirname(map_yaml), meta["image"])).convert("L"))
    res, (ox, oy) = float(meta["resolution"]), (float(meta["origin"][0]), float(meta["origin"][1]))
    h, w = img.shape
    free = img > 128

    def is_free(xy):
        col = ((xy[:, 0] - ox) / res).astype(int)
        row = (h - 1) - ((xy[:, 1] - oy) / res).astype(int)     # gym 은 맵 이미지를 상하 반전해 씀
        ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        out = np.zeros(len(xy), bool)
        out[ok] = free[row[ok], col[ok]]
        return out

    nrm = np.column_stack([-np.sin(heading), np.cos(heading)])
    dists = []
    for sign in (+1.0, -1.0):
        d = np.zeros(len(center))
        alive = np.ones(len(center), bool)                       # 아직 벽 못 만난 지점들
        for t in np.arange(step_m, max_search_m + step_m, step_m):
            probe = center + sign * t * nrm
            hit = alive & ~is_free(probe)
            d[hit] = t
            alive &= ~hit
            if not alive.any():
                break
        d[alive] = max_search_m                                  # 끝까지 벽 없으면 탐색 한계로
        dists.append(np.maximum(d - margin_m, 0.0))
    return dists[0], dists[1]


def _smooth_closed(x: np.ndarray, k: int) -> np.ndarray:
    """닫힌 배열 이동평균. 벽 거리가 픽셀 단위로 튀어서 테이프가 들쭉날쭉해지는 걸 막음."""
    if k < 2:
        return x
    ker = np.ones(k) / k
    return np.convolve(np.concatenate([x[-k:], x, x[:k]]), ker, mode="same")[k:-k]


def from_csv(path: str, cfg: Config, x_col: int = 1, y_col: int = 2, delimiter: str = ";") -> Track:
    raw = np.loadtxt(path, delimiter=delimiter, comments="#")
    step = cfg.lane.segment_len_m
    center = resample(raw[:, [x_col, y_col]], step)
    heading = _heading(center)
    s = np.arange(len(center)) * step
    length = len(center) * step

    if cfg.lane.follow_walls:                          # 맵 벽을 따라감. 폭이 구간마다 달라짐
        left, right = wall_offsets(center, heading, cfg.closed_loop.map_yaml, cfg.lane.wall_margin_m)
        k = max(1, int(round(0.5 / step)))             # 0.5 m 창
        left, right = _smooth_closed(left, k), _smooth_closed(right, k)
    else:                                              # 중심선에서 일정 폭. 실습실 테이프 트랙 방식
        half = cfg.lane.track_width_m / 2.0
        left = right = np.full(len(center), half)

    quads = np.concatenate([_tape_quads(center, heading, +left, cfg.lane.tape_width_m),
                            _tape_quads(center, heading, -right, cfg.lane.tape_width_m)])

    # 정답 경로를 무엇으로 볼 것인가. 테이프(모델이 보는 것)는 어느 쪽이든 똑같고 라벨만 달라짐
    if cfg.waypoints.line == "center":
        # 좌우 테이프의 중간선. 실차에서 HSV+IPM 으로 자동 라벨링할 때와 같은 기준
        nrm = np.column_stack([-np.sin(heading), np.cos(heading)])
        mid = center + ((left - right) / 2.0)[:, None] * nrm
        ref = resample(mid, step)
        ref_heading = _heading(ref)
        half = _interp_offsets((left + right) / 2.0, center, ref)     # 새 기준선에서의 좌우 여유
        return Track(ref, ref_heading, np.arange(len(ref)) * step, float(len(ref) * step),
                     quads, half, half)
    if cfg.waypoints.line != "racing":
        raise ValueError(f"waypoints.line must be 'center' or 'racing', got {cfg.waypoints.line!r}")
    # racing: CSV 에 든 레이싱 라인 그대로. 코너 안쪽 파고들어 center 보다 짧음
    return Track(center, heading, s, float(length), quads, left, right)


def _interp_offsets(values: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """src 위의 값을 dst 각 점에서 가장 가까운 src 점의 값으로 옮김."""
    i = np.argmin(((dst[:, None, :] - src[None, :, :]) ** 2).sum(-1), axis=1)
    return values[i]


def save(track: Track, path) -> None:
    np.savez_compressed(path, center=track.center, heading=track.heading, s=track.s,
                        length=track.length, quads=track.quads,
                        left_m=track.left_m, right_m=track.right_m)


def load(path) -> Track:
    d = np.load(path)
    return Track(d["center"], d["heading"], d["s"], float(d["length"]), d["quads"],
                 d["left_m"], d["right_m"])
