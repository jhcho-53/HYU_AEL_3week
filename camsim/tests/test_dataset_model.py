import copy
import numpy as np, torch, pytest
from camsim import config, track, dataset, model, gt
from camsim.render import bev_size

ARCHS = list(model.ARCHS)


@pytest.fixture(scope="module")
def ctx():
    cfg = config.load()
    cfg.model.pretrained = False            # 테스트에서 ImageNet 가중치 다운로드 안 함
    return cfg, track.from_csv("examples/example_waypoints.csv", cfg)


def cfg_with(cfg, arch):
    c = copy.deepcopy(cfg); c.model.arch = arch
    return c


def test_make_sample_is_bev_with_camera_mask(ctx):
    """모델 입력은 BEV. 카메라가 못 보는 근거리(0.32 m 안쪽)는 바닥색이어야 함 (실차 IPM 출력과 동일)."""
    cfg, trk = ctx
    bev, wp, pose, cam = dataset.make_sample(trk, cfg, np.random.default_rng(0), with_camera=True)
    h, w = bev_size(cfg)
    assert bev.shape == (h, w, 3) and cam.shape == (cfg.camera.image_height, cfg.camera.image_width, 3)
    assert wp.shape == (2,)
    near_rows = int((cfg.bev.x_range_m[1] - 0.30) / cfg.bev.resolution_m)      # x < 0.30 m -> 아래쪽 행들
    assert np.all(bev[near_rows:] == cfg.lane.color_floor)
    assert np.all(bev == cfg.lane.color_tape, axis=-1).sum() > 500

def test_dataset_yields_tensor_pairs(ctx):
    cfg, trk = ctx
    x, y = next(iter(dataset.SynthDataset(trk, cfg, seed=0)))
    assert x.shape == (3, *bev_size(cfg))
    assert x.dtype == torch.float32 and 0 <= x.min() <= x.max() <= 1
    assert y.shape == (2,)

def test_dataset_is_deterministic_per_seed(ctx):
    cfg, trk = ctx
    a = next(iter(dataset.SynthDataset(trk, cfg, seed=5)))
    b = next(iter(dataset.SynthDataset(trk, cfg, seed=5)))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

def test_dataloader_batches(ctx):
    cfg, trk = ctx
    dl = torch.utils.data.DataLoader(dataset.SynthDataset(trk, cfg), batch_size=4, num_workers=0)
    x, y = next(iter(dl))
    assert x.shape == (4, 3, *bev_size(cfg)) and y.shape == (4, 2)

@pytest.mark.parametrize("arch", ARCHS)
def test_model_forward(ctx, arch):
    cfg, _ = ctx
    net = model.WaypointNet(cfg_with(cfg, arch)).eval()
    assert net(torch.zeros(2, 3, *bev_size(cfg))).shape == (2, 2)

def test_unknown_arch_raises(ctx):
    cfg, _ = ctx
    with pytest.raises(ValueError, match="model.arch"):
        model.WaypointNet(cfg_with(cfg, "vgg"))

def test_constructor_leaves_bn_stats_alone(ctx):
    """FC 크기 재려고 흘린 0 입력이 BN running stats 를 건드리면 안 됨 (사전학습 통계가 오염됨)."""
    cfg, _ = ctx
    net = model.WaypointNet(cfg_with(cfg, "resnet18"))
    bn = net.backbone[1]
    assert torch.all(bn.running_mean == 0) and torch.all(bn.running_var == 1)

def test_resnet_takes_bgr_like_imagenet_takes_rgb(ctx, monkeypatch):
    """BGR 입력 + 뒤집은 conv1 이, 원래 ResNet 에 RGB + ImageNet 정규화를 넣은 것과 같아야 함.
    이게 틀리면 사전학습 효과가 조용히 사라짐."""
    import torchvision
    cfg, _ = ctx
    torch.manual_seed(0)
    ref = torchvision.models.resnet18(weights=None).eval()
    state = copy.deepcopy(ref.state_dict())
    real = torchvision.models.resnet18
    def fake(weights=None):
        m = real(weights=None); m.load_state_dict(state); return m
    monkeypatch.setattr(torchvision.models, "resnet18", fake)
    net = model.WaypointNet(cfg_with(cfg, "resnet18")).eval()

    x_bgr = torch.rand(1, 3, 64, 64)
    x_rgb = x_bgr.flip(1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    ref_backbone = torch.nn.Sequential(ref.conv1, ref.bn1, ref.relu, ref.maxpool,
                                       ref.layer1, ref.layer2, ref.layer3, ref.layer4)
    with torch.no_grad():
        ours = net.backbone((x_bgr - net.mean) / net.std)
        theirs = ref_backbone((x_rgb - mean) / std)
    assert torch.allclose(ours, theirs, atol=1e-5)

def test_predictor_shape(ctx):
    cfg, _ = ctx
    wp = model.Predictor(model.WaypointNet(cfg), cfg).predict(np.zeros((*bev_size(cfg), 3), np.uint8))
    assert wp.shape == (2,)

def test_oracle_matches_gt(ctx):
    cfg, trk = ctx
    pose = np.array([*trk.center[20], trk.heading[20]])
    o = model.OraclePredictor(trk, cfg)
    o.set_pose(pose)
    assert np.allclose(o.predict(None), gt.waypoint_ahead(pose, trk, cfg))

@pytest.mark.parametrize("arch", ARCHS)
def test_save_load(ctx, tmp_path, arch):
    cfg = cfg_with(ctx[0], arch)
    net = model.WaypointNet(cfg)
    model.save(net, tmp_path / "m.pt", cfg)
    net2 = model.load(tmp_path / "m.pt", cfg)
    x = torch.rand(1, 3, *bev_size(cfg))
    net.eval(); net2.eval()
    assert torch.allclose(net(x), net2(x))

@pytest.mark.parametrize("change", ["bev", "waypoint", "color", "arch"])
def test_load_rejects_changed_input_spec(ctx, tmp_path, change):
    """다른 규격·구조로 학습한 체크포인트는 조용히 로드되면 안 됨."""
    cfg, _ = ctx
    path = tmp_path / "m.pt"
    model.save(model.WaypointNet(cfg), path, cfg)
    changed = copy.deepcopy(cfg)
    if change == "bev": changed.bev.resolution_m *= 2
    elif change == "waypoint": changed.waypoints.ahead_m += 0.5
    elif change == "color": changed.lane.color_tape = [0, 0, 255]
    else: changed.model.arch = "small"
    with pytest.raises(ValueError, match="input_spec"):
        model.load(path, changed)

@pytest.mark.parametrize("arch", ARCHS)
def test_onnx_export_is_trt_friendly(ctx, tmp_path, arch):
    """젯슨 TensorRT 로 넘기는 ONNX. legacy exporter 로 변환되고, TRT 기본 연산만 나오고, 결과가 같아야 함."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    cfg = cfg_with(ctx[0], arch)
    net = model.WaypointNet(cfg).eval()
    path = model.export_onnx(net, cfg, tmp_path / "m.onnx")
    ops = {n.op_type for n in onnx.load(path).graph.node}
    assert ops <= {"Conv", "Relu", "MaxPool", "Add", "Sub", "Div", "Flatten", "Gemm", "Constant", "Identity"}, ops
    x = torch.rand(1, 3, *bev_size(cfg))
    with torch.no_grad():
        ref = net(x).numpy()
    out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"bev": x.numpy()})[0]
    assert np.allclose(out, ref, atol=1e-4)

def test_predict_camera_matches_predict_on_ipm(ctx):
    """실차 경로(카메라 -> IPM -> predict)는 같은 BEV 를 직접 넣은 것과 같아야 함."""
    from camsim import camera, render
    cfg, trk = ctx
    H_g2i, H_i2g = camera.build(cfg)
    p = model.Predictor(model.WaypointNet(cfg), cfg)
    cam = render.render(np.array([*trk.center[50], trk.heading[50]]), trk.quads, None, H_g2i, cfg)
    assert np.allclose(p.predict_camera(cam, H_i2g), p.predict(render.ipm_bev(cam, H_i2g, cfg)))
