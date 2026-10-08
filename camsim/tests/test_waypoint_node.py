"""ros2/waypoint_node.py without ROS: the car's BEV is the notebook's BEV, and its ONNX waypoint equals the torch one."""
import importlib.util
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from camsim import config, real, week3_lab as lab

ROOT = Path(__file__).parents[2]


def load_node():
    spec = importlib.util.spec_from_file_location("waypoint_node", ROOT / "ros2" / "waypoint_node.py")
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    return node


@pytest.fixture(scope="module")
def car_model(tmp_path_factory):
    """A car_model folder like real_ipm_lab 9장 writes: a small untrained net, the sample camera."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from camsim import handoff, model
    cfg = config.load()
    cfg.model.arch = "small"
    torch.manual_seed(0)
    net = model.WaypointNet(cfg, pretrained=False).eval()
    folder = tmp_path_factory.mktemp("car") / "car_model"
    onnx = model.export_onnx(net, cfg, folder.parent / "model.onnx")
    handoff.export_checkpoint(Path(onnx), folder, cfg, "test")
    K, _, size = real.load_ost(lab.SAMPLE / "ost.yaml")
    np.save(folder / "H_i2g.npy", real.camsim_h_i2g(K, size, cfg, lab.SAMPLE_PITCH_DEG, lab.SAMPLE_HEIGHT_M, 0.0))
    shutil.copy(lab.SAMPLE / "ost.yaml", folder / "ost.yaml")
    return folder, net, cfg


def test_bev_and_waypoint_match_the_notebook_and_torch(car_model):
    pytest.importorskip("rosbags")
    from camsim import model
    folder, net, cfg = car_model
    car = load_node().CarModel(folder, "cpu")
    frame = lab.read_frames(lab.SAMPLE / "run_train2_part1_2hz")[10]["bgr"]
    K, D, size = real.load_ost(folder / "ost.yaml")
    undistorted = cv2.remap(frame, *real.undistort_maps(K, D, size), interpolation=cv2.INTER_LINEAR)
    bev = car.bev(frame)
    assert np.array_equal(bev, real.bev_from_camera(undistorted, np.load(folder / "H_i2g.npy"), cfg))   # 6장 데이터셋 그림
    assert np.allclose(car.predict(bev), model.Predictor(net, cfg).predict(bev), atol=1e-4)
    assert car.draw(bev, car.predict(bev)).shape == bev.shape


def test_missing_or_altered_files_are_refused(car_model, tmp_path):
    folder, _, _ = car_model
    node = load_node()
    broken = tmp_path / "car_model"
    shutil.copytree(folder, broken)
    (broken / "H_i2g.npy").unlink()
    with pytest.raises(FileNotFoundError, match="H_i2g.npy"):
        node.CarModel(broken, "cpu")
    shutil.copy(folder / "H_i2g.npy", broken / "H_i2g.npy")
    with open(broken / "model.onnx", "ab") as stream:
        stream.write(b"\0")
    with pytest.raises(ValueError, match="checkpoint.json"):
        node.CarModel(broken, "cpu")


def test_an_image_of_another_size_is_refused(car_model):
    car = load_node().CarModel(car_model[0], "cpu")
    with pytest.raises(ValueError, match="ost.yaml"):
        car.bev(np.zeros((600, 960, 3), np.uint8))
