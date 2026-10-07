"""학습 루프와 오프라인 평가.

손실은 Huber(SmoothL1). waypoint 회귀라 가끔 튀는 라벨에 L2 보다 덜 흔들림.
타깃이 미터 단위라 오차가 1 m 안이면 L2 처럼 동작. L1 구간 쓰고 싶으면 beta 줄일 것.
"""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import Config
from .track import Track
from .dataset import SynthDataset, DiskDataset, make_sample
from . import model as M


def _summary(errs_m) -> dict:
    errs = np.asarray(errs_m, float)
    return {"mean_m": float(errs.mean()), "max_m": float(errs.max()), "n": int(len(errs)), "errs_m": errs}


def evaluate(predictor, track: Track, cfg: Config, n: int = 200, seed: int = 123, degrade_fn=None) -> dict:
    """새로 뽑은 pose n 개에 대한 waypoint 오차(m). 예측점과 정답점 사이 거리.

    degrade_fn(bev, rng) 주면 열화된 입력에 대한 강건성 측정이 됨 (노트북 3장 sim-to-real 표).
    """
    from .camera import build
    from .render import bev_visibility_mask
    rng = np.random.default_rng(seed)
    mask = bev_visibility_mask(build(cfg)[0], cfg)
    errs = []
    for _ in range(n):
        bev, wp, pose = make_sample(track, cfg, rng, degrade_fn, mask=mask)
        if hasattr(predictor, "set_pose"):        # OraclePredictor 는 이미지가 아니라 pose 를 봄
            predictor.set_pose(pose)
        errs.append(np.hypot(*(predictor.predict(bev) - wp)))
    return _summary(errs)


def _batches(dataset, batch_size, num_workers, seed):
    """무한 배치 스트림. DiskDataset 은 epoch 계속 돌려서 무한으로 만듦."""
    if isinstance(dataset, torch.utils.data.IterableDataset):
        yield from DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    else:
        g = torch.Generator().manual_seed(seed)
        while True:
            yield from DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  generator=g, drop_last=True)


@torch.no_grad()
def _val_loss(net, val_dataset, loss_fn, batch_size, val_batches, device):
    net.eval()
    dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    tot, k = 0.0, 0
    for x, y in dl:
        tot += loss_fn(net(x.to(device)), y.to(device)).item(); k += 1
        if k >= val_batches:
            break
    net.train()
    return tot / max(k, 1)


def train(track: Track, cfg: Config, steps: int, batch_size: int = 32, lr: float = 1e-3,
          device: str = "cpu", out_path=None, num_workers: int = 0, log_every: int = 50, seed: int = 0,
          dataset=None, val_dataset=None, val_batches: int = 8, callback=None):
    """dataset 안 주면 온더플라이(SynthDataset)로 학습.

    val_dataset 있으면 log_every 마다 val loss 도 재서 history 에 넣음.
    callback(history) 는 log_every 마다 불림. 노트북에서 loss 곡선 실시간으로 그릴 때 씀.
    """
    if dataset is not None and not isinstance(dataset, torch.utils.data.IterableDataset) \
            and len(dataset) < batch_size:
        # drop_last=True 라 배치가 하나도 안 나와서 무한루프 됨. 조용히 도는 것보다 여기서 죽는 게 나음
        raise ValueError(f"dataset has {len(dataset)} samples but batch_size is {batch_size}; "
                         f"generate more data or lower the batch size")
    torch.manual_seed(seed)
    net = M.WaypointNet(cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loss_fn = torch.nn.SmoothL1Loss()
    dl = _batches(dataset if dataset is not None else SynthDataset(track, cfg, seed=seed), batch_size, num_workers, seed)
    history, run, t0 = [], 0.0, time.time()
    net.train()
    for step, (x, y) in enumerate(dl, 1):
        x, y = x.to(device), y.to(device)
        loss = loss_fn(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run += loss.item()
        if step % log_every == 0:
            rec = {"step": step, "loss": run / log_every, "sec": time.time() - t0}
            if val_dataset is not None:
                rec["val_loss"] = _val_loss(net, val_dataset, loss_fn, batch_size, val_batches, device)
            history.append(rec)
            msg = f"step {step:6d}  loss {rec['loss']:.4f}" + (f"  val {rec['val_loss']:.4f}" if "val_loss" in rec else "")
            print(msg + f"  {rec['sec']:6.0f}s", flush=True)
            if callback is not None:
                callback(history)
            run = 0.0
        if step >= steps:
            break
    net.eval()
    if out_path is not None:
        M.save(net, out_path, cfg)
    return net, history


def evaluate_dataset(predictor, ds: DiskDataset, n: int = None) -> dict:
    """저장된 이미지(보통 val split)로 재는 waypoint 오차. 증강 없이 원본 씀."""
    n = len(ds) if n is None else min(n, len(ds))
    errs = [np.hypot(*(predictor.predict(ds.load_image(i)) - ds.wps[ds.idx[i]])) for i in range(n)]
    return _summary(errs)
