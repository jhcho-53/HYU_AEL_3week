import os, copy, numpy as np, torch, pytest, cv2
from camsim import config, track, dataset, train, model, gt

@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    cfg = config.load()
    cfg.model.arch, cfg.model.pretrained = "small", False   # 루프 검사라 빠른 모델로
    trk = track.from_csv(cfg.closed_loop.centerline_csv, cfg)
    root = str(tmp_path_factory.mktemp("ds"))
    dataset.generate_dataset(trk, cfg, 30, root, seed=1, log_every=0)
    return cfg, trk, root

def test_files_and_labels_written(ctx):
    cfg, trk, root = ctx
    files, poses, wps = dataset.read_labels(root)
    assert len(files) == 30 and poses.shape == (30, 3) and wps.shape == (30, 2)
    assert sorted(os.listdir(os.path.join(root, "images"))) == files
    from camsim.render import bev_size
    assert cv2.imread(os.path.join(root, "images", files[0])).shape == (*bev_size(cfg), 3)
    assert os.path.isfile(os.path.join(root, "spec.json"))

def test_labels_match_geometry(ctx):
    """저장된 waypoint 는 저장된 pose 에서 다시 계산한 GT 와 같아야 함 (라벨 파일이 자기 설명적)."""
    cfg, trk, root = ctx
    _, poses, wps = dataset.read_labels(root)
    for p, w in zip(poses[:5], wps[:5]):
        assert np.allclose(gt.waypoint_ahead(p, trk, cfg), w, atol=1e-3)

def test_split_is_disjoint_and_deterministic(ctx):
    cfg, trk, root = ctx
    tr = dataset.DiskDataset(root, cfg, "train", val_frac=0.2)
    va = dataset.DiskDataset(root, cfg, "val", val_frac=0.2)
    assert len(tr) == 24 and len(va) == 6
    assert not set(tr.idx) & set(va.idx)
    assert (dataset.split_indices(30, "val", 0.2) == va.idx).all()

def test_item_shapes_and_augment_toggle(ctx):
    cfg, trk, root = ctx
    ds = dataset.DiskDataset(root, cfg, "all")
    x, y = ds[0]
    from camsim.render import bev_size
    assert x.shape == (3, *bev_size(cfg)) and y.shape == (2,)
    x2, _ = ds[0]
    assert torch.equal(x, x2)                       # 증강 없으면 결정적
    assert torch.allclose(y * cfg.waypoints.norm_m, torch.from_numpy(ds.wps[0]).float())

def test_train_on_disk_and_evaluate(ctx, tmp_path):
    cfg, trk, root = ctx
    ds = dataset.DiskDataset(root, cfg, "train", val_frac=0.2)
    net, hist = train.train(trk, cfg, steps=12, batch_size=4, dataset=ds, log_every=6, out_path=tmp_path / "m.pt")
    assert len(hist) == 2 and (tmp_path / "m.pt").exists()
    r = train.evaluate_dataset(model.Predictor(net, cfg), dataset.DiskDataset(root, cfg, "val", val_frac=0.2))
    assert r["n"] == 6 and np.isfinite(r["mean_m"]) and r["errs_m"].shape == (6,)

def test_changed_config_is_detected(ctx):
    """설정 바뀌면 옛 데이터를 조용히 쓰면 안 됨. needs_regeneration 이 알려주고 DiskDataset 은 거부함."""
    cfg, trk, root = ctx
    assert not dataset.needs_regeneration(root, cfg, 30)
    assert dataset.needs_regeneration(root, cfg, 31)              # 장 수 바뀌어도 다시 만듦
    assert not dataset.needs_regeneration(root, cfg)              # n 안 주면 설정만 봄 (옛 노트북 호환)
    cfg2 = copy.deepcopy(cfg); cfg2.waypoints.ahead_m += 0.5
    assert dataset.needs_regeneration(root, cfg2, 30)
    with pytest.raises(ValueError, match="different config"):
        dataset.DiskDataset(root, cfg2)
    assert dataset.needs_regeneration(os.path.join(root, "nope"), cfg, 30)

def test_augment_fn_hook_applied_at_load(ctx):
    cfg, trk, root = ctx
    calls = []
    def fn(bev, rng):
        calls.append(1); out = bev.copy(); out[:] = 0; return out
    ds = dataset.DiskDataset(root, cfg, "all", augment_fn=fn)
    x, _ = ds[0]
    assert calls == [1] and float(x.max()) == 0.0
