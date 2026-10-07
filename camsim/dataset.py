"""학습 데이터. 모델 입력은 BEV, 라벨은 waypoint 하나의 (x, y).

  시뮬 : pose -> render_bev -> 카메라 가시 마스크 -> (증강)
  실차 : 카메라 -> undistort -> IPM

둘 다 config 의 bev 섹션(범위·해상도)을 공유하니까 같은 모델에 그대로 들어감.
실차에서 찍은 BEV 를 labels.csv 포맷으로 저장하면 여기 코드 그대로 쓸 수 있음.

Dataset 이 둘인데, SynthDataset 은 디스크 없이 매 샘플 새로 그리고, DiskDataset 은 미리 저장해 둔
폴더를 읽음. 노트북은 DiskDataset 쪽 (같은 데이터로 반복 학습해야 비교가 되니까).
"""
import csv
import json
import os
from dataclasses import asdict
import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from .config import Config
from .track import Track
from . import gt, render


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img_bgr)).permute(2, 0, 1).float().div_(255.0)


def make_sample(track: Track, cfg: Config, rng: np.random.Generator, augment_fn=None,
                mask: np.ndarray = None, with_camera: bool = False):
    """pose 하나 뽑아 (bev, waypoint (x, y) m, pose) 를 만듦.

    augment_fn : f(bev, rng) -> bev. 라벨은 증강과 무관한 참값이라, 증강은 "같은 정답을
                 다르게 본 것"이어야 함 (augment.py 맨 위 참고).
    mask       : bev_visibility_mask 결과. 안 주면 여기서 계산하는데 느림. 반복 호출할 거면 미리 만들 것.
    with_camera: 원근 카메라 뷰도 같이 렌더. 시각화 전용, 학습엔 안 씀.
    """
    from .camera import build
    H_g2i = build(cfg)[0]
    if mask is None:
        mask = render.bev_visibility_mask(H_g2i, cfg)
    pose = gt.sample_pose(track, cfg, rng)
    bev = render.render_bev(pose, track.quads, cfg, mask)
    if augment_fn is not None:
        bev = augment_fn(bev, rng)
    wp = gt.waypoint_ahead(pose, track, cfg)
    if with_camera:
        return bev, wp, pose, render.render(pose, track.quads, None, H_g2i, cfg)
    return bev, wp, pose


class SynthDataset(IterableDataset):
    """디스크 안 쓰고 매번 새로 그리는 무한 스트림."""

    def __init__(self, track: Track, cfg: Config, seed: int = 0, augment_fn=None):
        from .camera import build
        self.track, self.cfg, self.seed, self.augment_fn = track, cfg, seed, augment_fn
        self.mask = render.bev_visibility_mask(build(cfg)[0], cfg)

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid])      # worker 마다 다른 난수열
        norm = self.cfg.waypoints.norm_m
        while True:
            img, wp, _ = make_sample(self.track, self.cfg, rng, self.augment_fn, mask=self.mask)
            yield to_tensor(img), torch.from_numpy(wp / norm).float()


# ---- 디스크 데이터셋 -----------------------------------------------------------

LABELS_CSV = "labels.csv"
IMAGES_DIR = "images"
SPEC_JSON = "spec.json"
LABEL_HEADER = ["file", "x", "y", "theta", "wp_x", "wp_y"]


def dataset_spec(cfg: Config) -> dict:
    """생성된 이미지와 라벨에 영향 주는 설정 전부. 이게 바뀌었으면 데이터 다시 만들어야 함."""
    return {"camera": asdict(cfg.camera), "lane": asdict(cfg.lane), "bev": asdict(cfg.bev),
            "waypoints": asdict(cfg.waypoints), "sampling": asdict(cfg.sampling),
            "track": {"centerline_csv": cfg.closed_loop.centerline_csv,
                      "map_yaml": cfg.closed_loop.map_yaml}}


def needs_regeneration(out_dir: str, cfg: Config, n: int = None) -> bool:
    """labels.csv 없거나, 저장 당시 설정(과 n 을 주면 장 수)이 지금과 다르면 True."""
    if not os.path.isfile(os.path.join(out_dir, LABELS_CSV)):
        return True
    spec_path = os.path.join(out_dir, SPEC_JSON)
    if not os.path.isfile(spec_path):
        return True                                       # 옛 포맷. 뭘로 만든 건지 모름
    with open(spec_path, encoding="utf-8") as f:
        saved = json.load(f)
    if n is None:                                         # 옛 노트북은 n 을 안 넘김. 설정만 비교
        saved.pop("n", None)
        return saved != dataset_spec(cfg)
    return saved != {"n": int(n), **dataset_spec(cfg)}


def generate_dataset(track: Track, cfg: Config, n: int, out_dir: str, seed: int = 0,
                     augment_fn=None, log_every: int = 2000) -> str:
    """BEV n 장을 out_dir/images/*.png 와 labels.csv 로 저장하고 labels.csv 경로 반환.

    저장되는 건 증강 없는 원본. 증강은 보통 로딩 때 DiskDataset(augment_fn=...) 로 넣음.
    그래야 같은 데이터로 증강만 바꿔 가며 비교 가능. 같이 저장하는 spec.json 으로 나중에
    설정 바뀌었는지 알 수 있음.
    """
    from .camera import build
    img_dir = os.path.join(out_dir, IMAGES_DIR)
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    mask = render.bev_visibility_mask(build(cfg)[0], cfg)
    path = os.path.join(out_dir, LABELS_CSV)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LABEL_HEADER)
        for i in range(n):
            img, wp, pose = make_sample(track, cfg, rng, augment_fn, mask=mask)
            name = f"{i:06d}.png"
            cv2.imwrite(os.path.join(img_dir, name), img)
            w.writerow([name, *np.round(pose, 6), *np.round(wp, 4)])
            if log_every and (i + 1) % log_every == 0:
                print(f"{i + 1}/{n}", flush=True)
    with open(os.path.join(out_dir, SPEC_JSON), "w", encoding="utf-8") as f:
        json.dump({"n": int(n), **dataset_spec(cfg)}, f, ensure_ascii=False, indent=1)
    return path


def read_labels(out_dir: str):
    """labels.csv -> (파일명 리스트, poses (N,3), waypoints (N,2))."""
    with open(os.path.join(out_dir, LABELS_CSV), newline="") as f:
        rows = list(csv.reader(f))
    rows = rows[1:]
    files = [r[0] for r in rows]
    arr = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float64).reshape(len(rows), -1)
    if arr.shape[1] != 5:
        raise ValueError(f"labels.csv has {arr.shape[1]} value columns, expected 5 (x, y, theta, wp_x, wp_y)")
    return files, arr[:, :3], arr[:, 3:5]


def split_indices(n: int, split: str, val_frac: float = 0.1, seed: int = 0) -> np.ndarray:
    """seed 만 같으면 언제 불러도 같은 train/val 분리."""
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(round(n * val_frac))
    if split == "all":
        return np.arange(n)
    if split == "val":
        return np.sort(perm[:n_val])
    if split == "train":
        return np.sort(perm[n_val:])
    raise ValueError(f"split must be train/val/all, got {split!r}")


class DiskDataset(torch.utils.data.Dataset):
    """generate_dataset 이 만든 폴더를 읽음. __getitem__ -> (tensor (3,h,w), target (2,))."""

    def __init__(self, root: str, cfg: Config, split: str = "train", val_frac: float = 0.1,
                 seed: int = 0, augment_fn=None):
        """augment_fn: f(bev_bgr, rng) -> bev_bgr, 로딩 때 적용. None 이면 저장된 이미지 그대로."""
        self.root, self.cfg, self.augment_fn = root, cfg, augment_fn
        spec_path = os.path.join(root, SPEC_JSON)
        if os.path.isfile(spec_path):
            with open(spec_path, encoding="utf-8") as f:
                saved = json.load(f); saved.pop("n", None)        # 장 수는 학습엔 상관없음
                if saved != dataset_spec(cfg):
                    raise ValueError(f"{root} was generated with a different config; regenerate it")
        self.files, self.poses, self.wps = read_labels(root)
        self.idx = split_indices(len(self.files), split, val_frac, seed)
        self.seed = seed

    def __len__(self):
        return len(self.idx)

    def load_image(self, i: int) -> np.ndarray:
        """i 는 이 split 안의 인덱스."""
        img = cv2.imread(os.path.join(self.root, IMAGES_DIR, self.files[self.idx[i]]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.files[self.idx[i]])
        return img

    def __getitem__(self, i: int):
        img = self.load_image(i)
        if self.augment_fn is not None:
            # 같은 이미지라도 epoch 마다 다르게 증강되도록 torch 쪽 난수를 섞어 넣음
            rng = np.random.default_rng([self.seed, int(self.idx[i]), int(torch.randint(0, 2**31 - 1, (1,)))])
            img = self.augment_fn(img, rng)
        wp = self.wps[self.idx[i]]
        return to_tensor(img), torch.from_numpy(wp / self.cfg.waypoints.norm_m).float()
