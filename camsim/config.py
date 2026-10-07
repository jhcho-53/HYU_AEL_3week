"""config.yaml -> dataclass.

학생이 건드리는 파일이 config.yaml 하나뿐이라, 키를 빠뜨리거나 오타 내면 여기서 잡아줌.
그냥 통과시키면 한참 뒤 엉뚱한 데서 KeyError 나서 원인 찾기 어려움.
"""
from dataclasses import dataclass, fields
from typing import List, Optional
import os, yaml

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


class ConfigError(Exception):
    pass


@dataclass
class Camera:
    image_width: int
    image_height: int
    sensor_width_mm: float
    sensor_height_mm: float
    hfov_deg: float
    height_m: float
    pitch_deg: float
    offset_x_m: float
    h_i2g_file: Optional[str]


@dataclass
class Lane:
    follow_walls: bool
    wall_margin_m: float
    track_width_m: float
    tape_width_m: float
    segment_len_m: float
    color_floor: List[int]
    color_tape: List[int]


@dataclass
class Bev:
    x_range_m: List[float]
    y_range_m: List[float]
    resolution_m: float


@dataclass
class Render:
    near_m: float
    far_m: float
    lidar_fov_rad: float


@dataclass
class Waypoints:
    line: str
    ahead_m: float
    norm_m: float


@dataclass
class Sampling:
    lateral_frac: float
    heading_deg: float


@dataclass
class Augment:
    pitch_jitter_deg: float
    ipm_blur_max_px: int
    tape_dropout_prob: float
    brightness_delta: float
    contrast_range: List[float]
    gamma_range: List[float]
    hue_shift_deg: float
    sat_scale: List[float]
    illum_strength: float
    shadow_prob: float
    shadow_darkness: List[float]
    blur_max_px: int
    noise_sigma: float
    jpeg_quality: List[int]


@dataclass
class Model:
    arch: str
    pretrained: bool


@dataclass
class ClosedLoop:
    map_yaml: str
    centerline_csv: str
    control_hz: int
    speed_mps: float
    max_steps: int
    car_length_m: float
    car_width_m: float
    wheelbase_m: float
    steer_max_rad: float


@dataclass
class Config:
    camera: Camera
    lane: Lane
    bev: Bev
    render: Render
    waypoints: Waypoints
    sampling: Sampling
    augment: Augment
    model: Model
    closed_loop: ClosedLoop
    path: str = DEFAULT_PATH


def _build(cls, section: str, data):
    """yaml 한 섹션 -> dataclass. 빠진 키와 모르는 키를 한 번에 모아서 보고."""
    if not isinstance(data, dict):
        raise ConfigError(f"section '{section}' must be a mapping")
    names = {f.name for f in fields(cls)}
    missing = [f"{section}.{n}" for n in names if n not in data]
    extra = [f"{section}.{k}" for k in data if k not in names]
    if missing or extra:
        raise ConfigError(f"missing keys: {missing}; unknown keys: {extra}")
    return cls(**data)


def load(path: Optional[str] = None) -> Config:
    path = path or DEFAULT_PATH
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    sections = {f.name: f.type for f in fields(Config) if f.name != "path"}
    unknown = [k for k in raw if k not in sections]
    if unknown:
        raise ConfigError(f"unknown top-level keys: {unknown}")
    built = {}
    for name, cls in sections.items():
        if name not in raw:
            raise ConfigError(f"missing section '{name}'")
        built[name] = _build(cls, name, raw[name])
    # 렌더 해상도 비율이 센서 비율과 다르면 화각 계산이 조용히 틀어짐.
    # 기본값은 640x400 = 1.600, 6.62x4.14mm = 1.599 로 맞춰 둔 것.
    cam = built["camera"]
    render_ar = cam.image_width / cam.image_height
    sensor_ar = cam.sensor_width_mm / cam.sensor_height_mm
    if abs(render_ar - sensor_ar) > 0.02:
        raise ConfigError(
            f"camera aspect ratio mismatch: image {cam.image_width}x{cam.image_height} "
            f"(ratio {render_ar:.4f}) vs sensor {cam.sensor_width_mm}x{cam.sensor_height_mm}mm "
            f"(ratio {sensor_ar:.4f})"
        )
    return Config(path=path, **built)
