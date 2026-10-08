"""차에서 쓰는 waypoint 노드: 카메라 영상 -> BEV -> 모델 -> /waypoint (geometry_msgs/PointStamped).

real_ipm_lab 노트북과 같은 경로로 BEV 를 만듦 (ost.yaml 로 왜곡 보정 -> camsim 해상도 -> H_i2g 로 IPM).
그래서 노트북 9장이 내보낸 car_model 폴더(model.onnx, checkpoint.json, H_i2g.npy, ost.yaml)를 그대로 씀.
빌드 없이 레포 폴더에서 실행 (ros2/README.md):

    python3 ros2/waypoint_node.py --ros-args -p model_dir:=car_model

CarModel 은 ROS 없이도 돎 (테스트가 씀).
"""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # 레포 폴더의 camsim
from camsim import config, handoff, real, render                 # noqa: E402  (torch 는 안 씀)

MODEL_FILES = ("model.onnx", "checkpoint.json", "H_i2g.npy", "ost.yaml")


class CarModel:
    """car_model 폴더 하나로 영상 -> BEV -> waypoint (x, y) m (후륜축 기준, x 전방, y 왼쪽)."""

    def __init__(self, model_dir, device="cuda"):
        d = Path(model_dir).expanduser()
        missing = [name for name in MODEL_FILES if not (d / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{d} 에 {', '.join(missing)} 가 없음. real_ipm_lab 9장이 만든 car_model 폴더를 쓸 것.")
        manifest = json.loads((d / "checkpoint.json").read_text(encoding="utf-8"))
        if manifest.get("file") != "model.onnx" or handoff.sha256_file(d / "model.onnx") != manifest.get("sha256"):
            raise ValueError(f"{d / 'model.onnx'} 가 checkpoint.json 과 다름 (복사 중 깨졌거나 다른 모델). 9장 폴더를 다시 받을 것.")
        self.cfg = config.load()
        trained = manifest["config"]
        for key in ("bev", "waypoints"):
            if trained[key] != asdict(getattr(self.cfg, key)):
                raise ValueError(f"모델이 학습된 {key} 설정이 이 레포의 camsim/config.yaml 과 다름: {trained[key]}")
        if list(trained["lane"]["color_floor"]) != list(self.cfg.lane.color_floor):
            raise ValueError("모델이 학습된 BEV 바닥색(lane.color_floor)이 이 레포 설정과 다름.")
        K, D, self.size = real.load_ost(d / "ost.yaml")
        self.maps = real.undistort_maps(K, D, self.size)
        self.H = np.load(d / "H_i2g.npy")
        import onnxruntime as ort
        cuda = device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
        self.session = ort.InferenceSession(str(d / "model.onnx"),
                                            providers=(["CUDAExecutionProvider"] if cuda else []) + ["CPUExecutionProvider"])
        self.provider = self.session.get_providers()[0]
        self.input = self.session.get_inputs()[0].name

    def bev(self, bgr):
        """카메라 원본 BGR -> 모델 입력 BEV (380x300). real_ipm_lab 6장 데이터셋과 같은 그림."""
        if (bgr.shape[1], bgr.shape[0]) != tuple(self.size):
            raise ValueError(f"영상 {bgr.shape[1]}x{bgr.shape[0]} 이 ost.yaml 의 {self.size[0]}x{self.size[1]} 과 다름.")
        return real.bev_from_camera(cv2.remap(bgr, *self.maps, interpolation=cv2.INTER_LINEAR), self.H, self.cfg)

    def predict(self, bev):
        """BEV -> waypoint (x, y) m. 입력은 학습 때처럼 BGR 0~1 (정규화는 모델 안에 있음)."""
        x = np.ascontiguousarray(bev.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return self.session.run(None, {self.input: x})[0][0].astype(float) * self.cfg.waypoints.norm_m

    def draw(self, bev, wp):
        """BEV 위에 후륜축에서 ahead_m 원(하늘색)과 모델이 찍은 점(자홍), 숫자."""
        out = bev.copy()
        a = np.radians(np.arange(-90, 91, 3))
        ring = render.bev_pixels(self.cfg.waypoints.ahead_m * np.column_stack([np.cos(a), np.sin(a)]), self.cfg)
        cv2.polylines(out, [np.round(ring).astype(np.int32)], False, (255, 255, 0), 1, cv2.LINE_AA)
        render.draw_points_bev(out, wp, self.cfg, (255, 0, 255), 6)
        text = f"x {wp[0]:.2f}  y {wp[1]:+.2f} m"
        cv2.rectangle(out, (0, 0), (8 + 11 * len(text), 26), (0, 0, 0), -1)
        cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1, cv2.LINE_AA)
        return out


def main():
    import rclpy
    from geometry_msgs.msg import PointStamped
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    class WaypointNode(Node):
        def __init__(self):
            super().__init__("waypoint_node")
            p = {name: self.declare_parameter(name, default).value for name, default in (
                ("model_dir", "car_model"),                  # 레포 폴더 기준 상대 경로도 됨
                ("image_topic", "/flir_camera/image_raw"),
                ("waypoint_topic", "/waypoint"),
                ("bev_topic", "/waypoint_bev"),              # 모델이 본 BEV + 찍은 점. "" 이면 안 냄
                ("bev_hz", 5.0),
                ("device", "cuda"))}                         # onnxruntime CUDA 가 없으면 CPU 로 돎
            self.model = CarModel(p["model_dir"], p["device"])
            self.pub = self.create_publisher(PointStamped, p["waypoint_topic"], 1)
            self.bev_pub = self.create_publisher(Image, p["bev_topic"], 1) if p["bev_topic"] else None
            self.bev_period, self.last_bev, self.count = 1.0 / p["bev_hz"], 0.0, 0
            qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(Image, p["image_topic"], self.on_image, qos)
            self.get_logger().info(f"준비됨: 모델 {p['model_dir']}/model.onnx ({self.model.provider}), "
                                   f"영상 {p['image_topic']} -> {p['waypoint_topic']} "
                                   f"(후륜축에서 {self.model.cfg.waypoints.ahead_m:g} m 앞 점), BEV 화면 {p['bev_topic']}")

        def on_image(self, msg):
            try:
                bev = self.model.bev(real.decode_image(msg))
                wp = self.model.predict(bev)
            except Exception as exc:                    # 영상 하나가 이상해도 노드는 계속 돎
                self.get_logger().error(f"영상 처리 실패: {type(exc).__name__}: {exc}", throttle_duration_sec=5.0)
                return
            out = PointStamped()
            out.header.stamp, out.header.frame_id = msg.header.stamp, "rear_axle"
            out.point.x, out.point.y = float(wp[0]), float(wp[1])
            self.pub.publish(out)
            self.count += 1
            if self.count == 1:
                self.get_logger().info(f"첫 waypoint: x {wp[0]:.2f} m, y {wp[1]:+.2f} m")
            now = time.monotonic()
            if self.bev_pub is not None and now - self.last_bev >= self.bev_period:
                self.last_bev = now
                img = self.model.draw(bev, wp)
                view = Image()
                view.header = out.header
                view.height, view.width, view.encoding = img.shape[0], img.shape[1], "bgr8"
                view.step, view.data = img.shape[1] * 3, img.tobytes()
                self.bev_pub.publish(view)

    rclpy.init(args=sys.argv)
    try:
        node = WaypointNode()
    except (FileNotFoundError, ValueError) as exc:
        print(f"시작 못 함: {exc}", file=sys.stderr)
        rclpy.try_shutdown()
        sys.exit(1)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):   # Ctrl+C: 조용히 끝냄
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
