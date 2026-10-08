"""ros2/drive_node.py without ROS: it drives only while RB is held and both topics are fresh, else speed 0."""
import importlib.util
import math
from pathlib import Path

import pytest

from camsim import config
from camsim.pure_pursuit import pure_pursuit

ROOT = Path(__file__).parents[2]


def load_node():
    spec = importlib.util.spec_from_file_location("drive_node", ROOT / "ros2" / "drive_node.py")
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    return node


NODE = load_node()
CFG = config.load()
CL, FAR = CFG.closed_loop, 2 * CFG.waypoints.ahead_m
RB_ONLY = [0] * NODE.RB + [1] + [0] * 5


def ready(t=10.0, wp=(1.0, 0.3), rb=True):
    driver = NODE.Driver(0.5, CL.wheelbase_m, CL.steer_max_rad, FAR)
    driver.on_waypoint("rear_axle", *wp, t)
    driver.on_joy(RB_ONLY if rb else [0] * 11, t)
    return driver


def test_drives_with_the_simulator_pure_pursuit_while_rb_is_held():
    speed, steering, state = ready().command(10.1)
    assert speed == 0.5 and state.startswith("주행")
    assert steering == pure_pursuit((1.0, 0.3), CL.wheelbase_m, CL.steer_max_rad) > 0   # + = 왼쪽


def test_waits_without_rb():
    assert ready(rb=False).command(10.1)[:2] == (0.0, 0.0)
    driver = ready()
    driver.on_joy([0] * 11, 10.05)                     # RB 를 뗌
    assert driver.command(10.1)[:2] == (0.0, 0.0)
    driver.on_joy([1, 0, 0], 10.1)                     # 버튼 수가 RB 번호보다 적은 조이스틱
    assert driver.command(10.1)[:2] == (0.0, 0.0)


@pytest.mark.parametrize("now, expect", [(10.0 + NODE.WAYPOINT_TIMEOUT - 0.01, 0.5),
                                         (10.0 + NODE.WAYPOINT_TIMEOUT + 0.01, 0.0)])
def test_old_waypoint_stops(now, expect):
    driver = ready()
    driver.on_joy(RB_ONLY, now)                        # 조이스틱은 계속 옴
    speed, _, state = driver.command(now)
    assert speed == expect
    assert expect or "/waypoint 가 안 옴" in state


def test_old_joy_stops():
    driver = ready()
    now = 10.0 + NODE.JOY_TIMEOUT + 0.01
    driver.on_waypoint("rear_axle", 1.0, 0.0, now)     # 모델은 계속 옴, 조이스틱만 꺼짐
    speed, _, state = driver.command(now)
    assert speed == 0.0 and "/joy 가 안 옴" in state


def test_nothing_received_yet():
    driver = NODE.Driver(0.5, CL.wheelbase_m, CL.steer_max_rad, FAR)
    assert driver.command(0.0)[:2] == (0.0, 0.0)
    driver.on_joy(RB_ONLY, 0.0)
    assert "/waypoint 가 안 옴" in driver.command(0.0)[2]


@pytest.mark.parametrize("frame_id, x, y", [("base_link", 1.0, 0.0), ("rear_axle", -0.5, 0.0),
                                            ("rear_axle", math.nan, 0.0), ("rear_axle", 1.0, math.inf),
                                            ("rear_axle", 3.1, 0.4)])   # 젯슨에서 트랙이 아닌 장면에 나온 값
def test_bad_waypoint_stops(frame_id, x, y):
    driver = ready()
    driver.on_waypoint(frame_id, x, y, 10.05)
    assert driver.command(10.1)[:2] == (0.0, 0.0)


def test_two_publishers_stop_even_with_rb():
    speed, _, state = ready().command(10.1, NODE.crowded({"/waypoint": 1, "/joy": 2, "/drive": 1}))
    assert speed == 0.0 and "/joy 를 내는 노드가 2개" in state


def test_crowded_names_only_topics_with_two_or_more_publishers():
    assert NODE.crowded({"/waypoint": 1, "/joy": 1, "/drive": 1}) == ""
    assert NODE.crowded({"/waypoint": 0, "/joy": 0, "/drive": 1}) == ""       # 아직 못 찾은 것은 다른 검사가 막음
    text = NODE.crowded({"/waypoint": 2, "/joy": 1, "/drive": 3})
    assert "/waypoint 를 내는 노드가 2개" in text and "/drive 를 내는 노드가 3개" in text and "/joy" not in text


@pytest.mark.parametrize("value", [True, False, "0.5", "abc", None])
def test_speed_parameter_must_be_a_number(value):
    with pytest.raises(ValueError, match="speed"):
        NODE.parse_speed(value)                         # speed:=true 가 1.0 m/s 로 바뀌지 않게


def test_speed_parameter_takes_ints_and_floats():
    assert NODE.parse_speed(1) == 1.0 and NODE.parse_speed(0.5) == 0.5


@pytest.mark.parametrize("speed", [0.0, -0.5, NODE.MAX_SPEED + 0.01, 5.0, math.nan, math.inf])
def test_speed_out_of_range_is_refused(speed):
    with pytest.raises(ValueError, match="speed"):
        NODE.Driver(speed, CL.wheelbase_m, CL.steer_max_rad, FAR)


def test_steering_is_clipped_like_the_simulator():
    _, steering, _ = ready(wp=(0.2, 0.98)).command(10.1)
    assert steering == pytest.approx(CL.steer_max_rad)


def test_a_point_just_inside_twice_the_ring_still_drives():
    speed, _, _ = ready(wp=(FAR - 0.01, 0.0)).command(10.1)
    assert speed == 0.5
