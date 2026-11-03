import math
from dataclasses import dataclass, field

from .config_components import FlightControllerConfig, TetherModelConfig, DomainRandomizationConfig, TrajectoryConfig

@dataclass(frozen=True)
class Termination:
    episode_length_s: float = 30.0
    terminate_if_payload_tilt_greater_than: float = math.pi / 4.0
    terminate_if_swingangle_greater_than: float = math.radians(60.0)
    terminate_if_x_greater_than: float = 3.0 # on 0-idx pl pos err (bbox around the moving reference, heading frame)
    terminate_if_y_greater_than: float = 3.0
    terminate_if_z_greater_than: float = 3.0
    terminate_if_payload_below_z: float = 0.2 # ground crash

@dataclass(frozen=True)
class Rewards:
    reward_terms: tuple[str,...] = ("pos_track", "vel_track", "swing_energy", "actions", "crash")

    sigma_pos_track_close: float = 0.2
    sigma_pos_track_far: float = 3.0
    sigma_vel_track: float = 0.5
    
    w_pos_track: float = 1.0
    w_vel_track: float = 1.0
    w_swing_energy: float = -1.0
    w_action: float = -0.1
    w_crash: float = -10.0

@dataclass(frozen=True)
class Observations:
    scale_pl_p: float = 1.0 / 3.0
    scale_pl_v: float = 1.0 / 6.0
    scale_pl_sa: float = 1.0 / math.radians(60.0)
    scale_pl_sr: float = 1.0 / math.pi
    scale_pl_br: float = 1.0 / math.pi
    scale_d_br: float = 1.0 / math.pi

@dataclass(frozen=True)
class EnvConfig:
    # Components configs
    fc_cfg: FlightControllerConfig = field(default_factory=FlightControllerConfig)
    tether_cfg: TetherModelConfig = field(default_factory=TetherModelConfig)
    dr_cfg: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    traj_cfg: TrajectoryConfig = field(default_factory=TrajectoryConfig)

    sim_dt: float = 0.005 # physics rate
    decimation: int = 4
    policy_dt: float = field(init=False) # policy rate (sim_dt * decimation)
    torch_compile: bool = True # fuse per-sim-step torch code

    num_actions: int = 4
    clip_actions: float = 1.0
    num_obs: int = field(init=False)
    num_priv_obs: int = 8 # Note: in addition to num_obs
    preview_points_s: tuple[float, ...] = (0.0, 0.3, 0.8, 1.5, 2.0) # Should start at 0.0 for current time track (K-dim)

    payload_init_altitude: float = 5.0

    obs: Observations = field(default_factory=Observations)
    rewards: Rewards = field(default_factory=Rewards)
    episode_termination: Termination = field(default_factory=Termination)
    succes_mean_track_err = 0.25
    succes_interval = 1200

    def __post_init__(self):
        object.__setattr__(self, "policy_dt", self.sim_dt * self.decimation)
        object.__setattr__(self, "num_obs", 20 + 6*len(self.preview_points_s))
        object.__setattr__(self.traj_cfg, "traj_max_duration", self.episode_termination.episode_length_s * 0.9)