"""gym 안에서 렌더 -> 추론 -> pure pursuit -> step 을 한 루프로 돌림. ROS 는 안 씀.

종료 사유(Result.reason)
  lap          한 바퀴 완주
  tape_crossed 차체 모서리가 테이프 안쪽 선을 넘음. 실차와 같은 실격 규칙
  collision    gym 벽 충돌. 이 맵은 벽이 테이프보다 훨씬 멀어서 거의 안 남
  max_steps    시간 초과

실차 트랙엔 벽이 없고 테이프가 경계라서, 실격 판정도 벽이 아니라 테이프로 함.
"""
from dataclasses import dataclass
import os
import warnings
import cv2
import numpy as np
from .config import Config
from .track import Track
from . import gt, render, viz
from .pure_pursuit import pure_pursuit


def make_env(cfg: Config):
    import gym
    from f110_gym.envs.base_classes import Integrator
    base = os.path.splitext(cfg.closed_loop.map_yaml)[0]
    with warnings.catch_warnings():
        # gym 이 RK4 고를 때마다 경고 뱉는데, 우리가 일부러 고른 거라 조용히 시킴
        warnings.filterwarnings("ignore", message="Chosen integrator is RK4.*")
        return gym.make("f110_gym:f110-v0", map=base, map_ext=".png", num_agents=1,
                        timestep=0.01, integrator=Integrator.RK4)


def pose_of(obs) -> np.ndarray:
    return np.array([obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]])


@dataclass
class Result:
    finished: bool
    reason: str
    steps: int
    mean_lateral_m: float
    max_lateral_m: float
    progress_m: float
    control_hz_eff: float
    time_s: float = 0.0                # 주행 시간(초). 랩타임 비교용
    lateral_trace: np.ndarray = None   # 제어 틱마다의 횡오차(m). 그래프용
    pose_trace: np.ndarray = None      # (steps,3) 제어 틱마다의 world pose. 맵에 경로 그릴 때


def _unwrap_progress(track: Track, s_prev: float, s_now: float) -> float:
    # 결승선 넘어가면 s 가 length -> 0 으로 점프함. 그 점프를 정상적인 전진으로 되돌림
    ds = s_now - s_prev
    if ds < -track.length / 2: ds += track.length
    if ds > track.length / 2: ds -= track.length
    return ds


def run(env, predictor, track: Track, cfg: Config, H_g2i: np.ndarray, start_index: int = 0,
        video_path=None) -> Result:
    cl = cfg.closed_loop

    # 제어 한 틱마다 물리를 몇 번 돌릴지. control_hz 가 물리 주파수의 약수가 아니면 반올림되면서
    # 실제 제어 주기가 요청값과 달라지므로, 그 경우 경고하고 실제값(hz_eff)을 결과에 남김
    physics_per_tick = max(1, int(round(1.0 / (env.timestep * cl.control_hz))))
    hz_eff = 1.0 / (env.timestep * physics_per_tick)
    if abs(hz_eff - cl.control_hz) / cl.control_hz > 0.02:
        warnings.warn(
            f"camsim: control_hz={cl.control_hz} is not an integer divisor of the gym physics "
            f"rate (1/{env.timestep}); effective control rate is {hz_eff:.3f} Hz instead",
            UserWarning,
        )

    p0 = track.center[start_index]
    obs, _, done, _ = env.reset(np.array([[p0[0], p0[1], track.heading[start_index]]]))

    mask = render.bev_visibility_mask(H_g2i, cfg)
    writer = None            # 영상은 [카메라 뷰 | 모델 입력 BEV]. 첫 프레임 크기로 열림

    lats, traveled, reason = [], 0.0, "max_steps"
    poses = []
    s_prev = track.s[gt.nearest_index(track, p0)]
    steps = 0
    try:
        for steps in range(1, cl.max_steps + 1):
            pose = pose_of(obs)
            bev = render.render_bev(pose, track.quads, cfg, mask)
            if hasattr(predictor, "set_pose"):
                predictor.set_pose(pose)
            wp = predictor.predict(bev)
            steer = pure_pursuit(wp, cl.wheelbase_m, cl.steer_max_rad)
            if video_path is not None:
                cam = render.draw_points(render.render(pose, track.quads, obs["scans"][0], H_g2i, cfg), wp, H_g2i)
                frame = viz.side_by_side(cam, render.draw_points_bev(bev.copy(), wp, cfg))
                if writer is None:
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), hz_eff,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            for _ in range(physics_per_tick):
                obs, _, done, _ = env.step(np.array([[steer, cl.speed_mps]]))
                if done:
                    break
            pose = pose_of(obs)
            lat = gt.lateral_error(track, pose[:2])
            lats.append(lat)
            poses.append(pose)
            s_now = track.s[gt.nearest_index(track, pose[:2])]
            traveled += _unwrap_progress(track, s_prev, s_now)
            s_prev = s_now
            if obs["collisions"][0]:
                reason = "collision"; break
            if gt.crosses_tape(pose, track, cfg):
                reason = "tape_crossed"; break
            if traveled >= track.length:
                reason = "lap"; break
    finally:
        if writer is not None:                     # 중간에 터져도 mp4 는 닫아야 재생됨
            writer.release()
    lats = np.array(lats) if lats else np.zeros(1)
    return Result(reason == "lap", reason, steps, float(lats.mean()), float(lats.max()), float(traveled),
                  hz_eff, steps / hz_eff, lats, np.array(poses).reshape(-1, 3))
