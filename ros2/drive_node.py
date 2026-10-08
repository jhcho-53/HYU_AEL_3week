"""차에서 쓰는 주행 노드: /waypoint (geometry_msgs/PointStamped) -> pure pursuit -> /drive (AckermannDriveStamped).

waypoint_node.py 가 찍은 점으로 가는 조향각을 2주차 시뮬레이터와 같은 camsim/pure_pursuit.py 로 계산하고,
속도는 speed 파라미터 값으로 고정해서 차량 스택의 자율주행 입력(/drive)으로 보냄.
조이스틱 RB 를 누르고 있는 동안만 달림. 아래 경우에는 언제나 속도 0 을 보냄:
  - RB 를 안 누름, 또는 /joy 가 0.5초 넘게 안 옴 (조이스틱이 꺼졌거나 차량 스택이 꺼짐)
  - /waypoint 가 0.25초 넘게 안 옴 (waypoint_node 나 카메라가 멈춤)
  - 모델이 찍은 점이 차 앞이 아니거나 1 m 원의 두 배보다 멂 (차선이 아닌 것을 보고 헷갈린 것)
  - /joy, /waypoint, /drive 중 하나라도 내는 노드가 2개 이상 (같은 노드를 두 번 켰거나 같은 와이파이의 다른 차)
빌드 없이 레포 폴더에서 실행 (ros2/README.md):

    python3 ros2/drive_node.py --ros-args -p speed:=0.5

Driver 는 ROS 없이도 돎 (테스트가 씀).
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # 레포 폴더의 camsim
from camsim import config                                        # noqa: E402
from camsim.pure_pursuit import pure_pursuit                     # noqa: E402

MAX_SPEED = 2.0          # m/s. 0.5 를 5 로 잘못 쳐도 안 켜지게
WAYPOINT_TIMEOUT = 0.25  # s. waypoint_node 는 약 20 Hz 로 냄
JOY_TIMEOUT = 0.5        # s. joy_node 는 버튼을 안 눌러도 20 Hz 로 냄
RB = 5                   # F710 RB 버튼 번호. 차량 스택 joy_teleop 의 autonomous_control 과 같은 버튼
CONTROL_HZ = 25.0        # 시뮬레이터 closed_loop.control_hz 와 같음


def parse_speed(value):
    """speed 파라미터 -> float. speed:=true 도 1.0 이 되어 버리므로 숫자만 받음."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"speed={value!r}: 숫자로 줄 것 (m/s, 예: 0.5)")
    return float(value)


def crowded(counts):
    """{토픽: 그 토픽을 내는 노드 수} -> 2개 이상인 토픽 설명. 다 1개 이하면 ""."""
    many = [f"{topic} 를 내는 노드가 {n}개" for topic, n in counts.items() if n > 1]
    return ", ".join(many) + ". 같은 노드를 두 번 켰거나, 같은 와이파이의 다른 차와 섞인 것" if many else ""


class Driver:
    """언제 달리고 어디로 꺾을지 정함. 시간은 time.monotonic() 초, 받은 시각으로 오래됐는지 봄."""

    def __init__(self, speed, wheelbase, steer_max, max_dist, frame_id="rear_axle"):
        if not (math.isfinite(speed) and 0 < speed <= MAX_SPEED):
            raise ValueError(f"speed={speed}: 0 보다 크고 {MAX_SPEED} 이하로 줄 것 (m/s, 예: 0.5)")
        self.speed, self.wheelbase, self.steer_max, self.frame_id = speed, wheelbase, steer_max, frame_id
        self.max_dist = max_dist     # 모델은 1 m 원 위의 점을 찍도록 배웠음. 이보다 멀면 헷갈린 것
        self.waypoint = None         # (frame_id, x, y, 받은 시각)
        self.joy = None              # (RB 누름, 받은 시각)

    def on_waypoint(self, frame_id, x, y, now):
        self.waypoint = (frame_id, x, y, now)

    def on_joy(self, buttons, now):
        self.joy = (len(buttons) > RB and buttons[RB] == 1, now)

    def command(self, now, busy=""):
        """-> (속도, 조향각, 상태). 상태 글은 바뀔 때만 로그로 찍음. busy 는 crowded() 가 낸 설명."""
        if busy:
            return 0.0, 0.0, f"정지: {busy}"
        if self.waypoint is None or now - self.waypoint[3] > WAYPOINT_TIMEOUT:
            return 0.0, 0.0, "정지: /waypoint 가 안 옴. 카메라와 waypoint_node 가 켜져 있는지 볼 것"
        frame_id, x, y, _ = self.waypoint
        if frame_id != self.frame_id:
            return 0.0, 0.0, f"정지: /waypoint 의 frame_id 가 {frame_id!r} 임 ({self.frame_id!r} 이어야 함)"
        if not (math.isfinite(x) and math.isfinite(y) and x > 0 and math.hypot(x, y) <= self.max_dist):
            return 0.0, 0.0, "정지: 모델이 1 m 앞이 아닌 엉뚱한 점을 냄. 카메라가 트랙을 보고 있는지 볼 것"
        if self.joy is None or now - self.joy[1] > JOY_TIMEOUT:
            return 0.0, 0.0, "정지: /joy 가 안 옴. 조이스틱 전원과 차량 스택이 켜져 있는지 볼 것"
        if not self.joy[0]:
            return 0.0, 0.0, "대기: RB 를 누르고 있으면 달림"
        return self.speed, pure_pursuit((x, y), self.wheelbase, self.steer_max), \
            f"주행: {self.speed:g} m/s. RB 를 떼면 멈춤"


def main():
    import signal
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import PointStamped
    from rcl_interfaces.msg import ParameterDescriptor
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import Joy

    class DriveNode(Node):
        def __init__(self):
            super().__init__("drive_node")
            speed = parse_speed(self.declare_parameter(                  # m/s, 고정. 2.0 까지. speed:=1 도 받음
                "speed", 0.5, ParameterDescriptor(dynamic_typing=True)).value)
            p = {name: self.declare_parameter(name, default).value for name, default in (
                ("waypoint_topic", "/waypoint"),
                ("joy_topic", "/joy"),
                ("drive_topic", "/drive"))}                              # 차량 스택 ackermann_mux 의 자율주행 입력
            cfg = config.load()                                          # 축간거리, 조향 한계는 시뮬레이터와 같음
            cl = cfg.closed_loop
            self.driver = Driver(speed, cl.wheelbase_m, cl.steer_max_rad, 2 * cfg.waypoints.ahead_m)
            self.watch = (p["waypoint_topic"], p["joy_topic"], p["drive_topic"])   # /drive 는 이 노드 하나만 내야 함
            self.state, self.started = None, time.monotonic()
            self.pub = self.create_publisher(AckermannDriveStamped, p["drive_topic"], 1)
            self.create_subscription(PointStamped, p["waypoint_topic"], self.on_waypoint, 1)
            self.create_subscription(Joy, p["joy_topic"], self.on_joy, 1)
            self.create_timer(1.0 / CONTROL_HZ, self.control)
            self.get_logger().info(f"준비됨: {p['waypoint_topic']} -> {p['drive_topic']}, 속도 {speed:g} m/s, "
                                   f"축간거리 {cl.wheelbase_m:g} m. 조이스틱 RB 를 누르고 있는 동안만 달림")

        def on_waypoint(self, msg):
            self.driver.on_waypoint(msg.header.frame_id, msg.point.x, msg.point.y, time.monotonic())

        def on_joy(self, msg):
            self.driver.on_joy(msg.buttons, time.monotonic())

        def control(self):
            busy = crowded({topic: self.count_publishers(topic) for topic in self.watch})
            speed, steering, state = self.driver.command(time.monotonic(), busy)
            self.publish(speed, steering)
            if state != self.state and time.monotonic() - self.started > 2.0:   # 켜자마자는 토픽이 아직 안 붙음
                self.state = state
                self.get_logger().info(state)

        def publish(self, speed, steering):
            msg = AckermannDriveStamped()
            msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), "base_link"
            msg.drive.speed, msg.drive.steering_angle = float(speed), float(steering)
            self.pub.publish(msg)

        def stop(self):
            self.publish(0.0, 0.0)
            self.pub.wait_for_all_acked(Duration(seconds=0.5))          # 받는 쪽이 받기 전에 끝나지 않게
            self.get_logger().info("끔: 마지막으로 속도 0 을 보냄")

    def interrupt(signum, frame):            # 터미널을 닫거나 kill 해도 Ctrl+C 처럼 끝냄
        raise KeyboardInterrupt

    # Ctrl+C 를 rclpy 가 먼저 받으면 통신이 먼저 닫혀서 마지막 속도 0 을 못 보냄. 그래서 Python 이 받게 함
    rclpy.init(args=sys.argv, signal_handler_options=SignalHandlerOptions.NO)
    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, interrupt)
    try:
        node = DriveNode()
    except ValueError as exc:
        print(f"시작 못 함: {exc}", file=sys.stderr)
        rclpy.try_shutdown()
        sys.exit(1)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):    # 끄는 동안 한 번 더 눌러도 0 은 보냄
            signal.signal(s, signal.SIG_IGN)
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
