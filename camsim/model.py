"""waypoint CNN 과 predict 래퍼.

시뮬 폐루프와 실차가 똑같이 Predictor.predict(bev) 를 부름. 그게 이 설계의 목표.
출력은 waypoint 하나의 (x, y), 단위 m.

젯슨에선 TensorRT 로 돌리므로 ONNX 로 깨끗하게 넘어가는 연산만 씀.
AdaptiveAvgPool 은 입력 크기가 출력 크기로 안 나눠떨어지면 legacy ONNX export 가 실패해서 안 씀.
"""
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn
from .config import Config
from .track import Track
from . import gt
from .dataset import to_tensor
from .render import ipm_bev, bev_size

ARCHS = ("resnet18", "small")

# ImageNet 정규화 값. 우리 입력은 OpenCV BGR 이라 순서도 BGR
_MEAN_BGR = (0.406, 0.456, 0.485)
_STD_BGR = (0.225, 0.224, 0.229)


def _small_backbone():
    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128), block(128, 128)), 128


def _resnet18_backbone(pretrained: bool):
    import torchvision
    r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    # ImageNet 가중치는 RGB 입력으로 학습됨. 입력 채널 순서를 뒤집어서 BGR 을 그대로 받게 함 (연산 추가 없음)
    with torch.no_grad():
        r.conv1.weight.copy_(r.conv1.weight[:, [2, 1, 0]])
    return nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4), 512


class WaypointNet(nn.Module):
    """backbone -> 1x1 conv -> flatten -> FC -> (x, y).

    pooling 대신 flatten 이라 feature map 의 칸(380x300 입력이면 12x10) 위치가 그대로 FC 에 들어감.
    대신 FC 크기가 BEV 크기에 묶임. 다른 BEV 로 학습한 체크포인트는 어차피 load() 가 거부함.
    """

    def __init__(self, cfg: Config, pretrained: bool = None):
        super().__init__()
        arch = cfg.model.arch
        if arch not in ARCHS:
            raise ValueError(f"model.arch must be one of {ARCHS}, got {arch!r}")
        if pretrained is None:
            pretrained = cfg.model.pretrained
        self.backbone, c = _resnet18_backbone(pretrained) if arch == "resnet18" else _small_backbone()
        self.register_buffer("mean", torch.tensor(_MEAN_BGR).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD_BGR).view(1, 3, 1, 1))

        # FC 입력 크기를 재려고 한 번 흘려 봄. train 모드면 BN 통계가 0 입력으로 오염되니 eval 로
        self.backbone.eval()
        with torch.no_grad():
            fh, fw = self.backbone(torch.zeros(1, 3, *bev_size(cfg))).shape[2:]
        self.backbone.train()
        self.head = nn.Sequential(nn.Conv2d(c, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                  nn.Flatten(), nn.Linear(32 * fh * fw, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, 2))

    def forward(self, x):
        """x: BGR 0~1, (N, 3, H, W)."""
        return self.head(self.backbone((x - self.mean) / self.std))


class Predictor:
    def __init__(self, net: nn.Module, cfg: Config, device: str = "cpu"):
        self.net, self.cfg, self.device = net.to(device).eval(), cfg, device

    @torch.no_grad()
    def predict(self, bev_bgr: np.ndarray) -> np.ndarray:
        """BEV -> waypoint (x, y) m. 시뮬과 실차가 공유하는 유일한 인터페이스."""
        x = to_tensor(bev_bgr)[None].to(self.device)
        return self.net(x)[0].cpu().numpy() * self.cfg.waypoints.norm_m

    def predict_camera(self, cam_bgr: np.ndarray, H_i2g: np.ndarray) -> np.ndarray:
        """실차용. 카메라 영상을 IPM 으로 펴서 predict 에 넘김."""
        return self.predict(ipm_bev(cam_bgr, H_i2g, self.cfg))


class OraclePredictor:
    """모델 대신 정답을 그대로 돌려줌. 학습된 모델 없이 폐루프를 돌려 보거나,
    모델 주행과 나란히 그려서 "모델 탓인지 원래 무리인 코너인지" 가를 때 씀."""

    def __init__(self, track: Track, cfg: Config, noise_sigma: float = 0.0, rng=None):
        self.track, self.cfg, self.sigma = track, cfg, noise_sigma
        self.rng = rng or np.random.default_rng(0)
        self.pose = None

    def set_pose(self, pose):
        self.pose = np.asarray(pose, float)

    def predict(self, img_bgr) -> np.ndarray:
        wp = gt.waypoint_ahead(self.pose, self.track, self.cfg)
        if self.sigma > 0:
            wp = wp + self.rng.normal(0, self.sigma, wp.shape)
        return wp


def _input_spec(cfg: Config) -> dict:
    """체크포인트가 어떤 입력/출력 규격·구조로 학습됐는지. 이게 다르면 가중치 이어 쓸 수 없음."""
    return {"bev": asdict(cfg.bev), "waypoints": asdict(cfg.waypoints),
            "lane_colors": {"floor": cfg.lane.color_floor, "tape": cfg.lane.color_tape},
            "arch": cfg.model.arch}


def save(net: nn.Module, path, cfg: Config) -> None:
    torch.save({"state_dict": net.state_dict(), "input_spec": _input_spec(cfg)}, path)


def load(path, cfg: Config) -> WaypointNet:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("input_spec") != _input_spec(cfg):
        raise ValueError("checkpoint input_spec does not match BEV, waypoint, lane color or arch config")
    net = WaypointNet(cfg, pretrained=False)        # 가중치는 체크포인트에서 옴. ImageNet 다운로드 불필요
    net.load_state_dict(ck["state_dict"])
    return net


def export_onnx(net: nn.Module, cfg: Config, path) -> str:
    """젯슨 TensorRT 용 ONNX. 입력 (1, 3, H, W) BGR 0~1 고정, 출력 (1, 2) = (x, y) / norm_m.

    legacy exporter 를 씀. TRT 튜토리얼·trtexec 가 가정하는 방식이고, 이 모델에선 Conv/Relu/MaxPool/Add/
    Sub/Div/Flatten/Gemm 만 나옴 (BN 은 conv 에 흡수됨).
    """
    net = net.eval().cpu()
    x = torch.zeros(1, 3, *bev_size(cfg))
    torch.onnx.export(net, x, str(path), input_names=["bev"], output_names=["wp"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    return str(path)
