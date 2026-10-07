import copy
import numpy as np, pytest
from camsim import config, track, camera, model, closed_loop as cl

@pytest.fixture(scope="module")
def ctx():
    cfg = config.load()
    cfg.closed_loop.max_steps = 600
    trk = track.from_csv(cfg.closed_loop.centerline_csv, cfg)
    env = cl.make_env(cfg)
    return cfg, trk, env, camera.build(cfg)[0]

def test_oracle_stays_on_track(ctx):
    cfg, trk, env, H = ctx
    r = cl.run(env, model.OraclePredictor(trk, cfg), trk, cfg, H)
    assert r.reason == "max_steps"
    assert r.max_lateral_m < 0.25
    assert r.progress_m > 20.0          # 600 ticks at ~25 Hz, 2 m/s -> tens of meters

def test_huge_noise_leaves_track(ctx):
    cfg, trk, env, H = ctx
    r = cl.run(env, model.OraclePredictor(trk, cfg, noise_sigma=2.0), trk, cfg, H)
    assert r.reason in ("tape_crossed", "collision") and not r.finished

def test_video_written(ctx, tmp_path):
    cfg, trk, env, H = ctx
    cfg2 = copy.deepcopy(cfg)
    cfg2.closed_loop.max_steps = 30
    p = tmp_path / "run.mp4"
    cl.run(env, model.OraclePredictor(trk, cfg2), trk, cfg2, H, video_path=p)
    assert p.exists() and p.stat().st_size > 1000

def test_control_hz_eff_matches_when_evenly_divisible(ctx):
    cfg, trk, env, H = ctx
    cfg2 = copy.deepcopy(cfg)
    cfg2.closed_loop.control_hz = 25    # 100 Hz physics / 25 -> exact 4 steps/tick
    cfg2.closed_loop.max_steps = 5
    r = cl.run(env, model.OraclePredictor(trk, cfg2), trk, cfg2, H)
    assert r.control_hz_eff == pytest.approx(25.0)

def test_control_hz_warns_when_not_evenly_divisible(ctx):
    cfg, trk, env, H = ctx
    cfg2 = copy.deepcopy(cfg)
    cfg2.closed_loop.control_hz = 30    # 100/30 rounds to 3 steps/tick -> 33.3 Hz effective
    cfg2.closed_loop.max_steps = 5
    with pytest.warns(UserWarning, match="camsim: control_hz"):
        r = cl.run(env, model.OraclePredictor(trk, cfg2), trk, cfg2, H)
    assert r.control_hz_eff == pytest.approx(100 / 3)


def test_lateral_trace_recorded(ctx):
    import copy
    cfg, trk, env, H = ctx
    cfg2 = copy.deepcopy(cfg); cfg2.closed_loop.max_steps = 20
    r = cl.run(env, model.OraclePredictor(trk, cfg2), trk, cfg2, H)
    assert r.lateral_trace.shape == (r.steps,) and r.lateral_trace.max() == r.max_lateral_m


def test_pose_trace_recorded(ctx):
    import copy
    cfg, trk, env, H = ctx
    cfg2 = copy.deepcopy(cfg); cfg2.closed_loop.max_steps = 15
    r = cl.run(env, model.OraclePredictor(trk, cfg2), trk, cfg2, H)
    assert r.pose_trace.shape == (r.steps, 3)
    assert np.hypot(*(r.pose_trace[-1, :2] - r.pose_trace[0, :2])) > 0.3
