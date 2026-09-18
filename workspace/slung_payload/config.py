import os
from dataclasses import dataclass, asdict, is_dataclass, field

EPS = 1e-9
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DRONE_PATH = os.path.join(REPO_ROOT, "system_model", "Starling2MaxURDF", "model", "starling2max.urdf")
PAYLOAD_PATH = os.path.join(REPO_ROOT, "system_model", "payload", "box", "box.urdf")

def dataclass_to_dict(obj):
    assert is_dataclass(obj) and not isinstance(obj, type), f"{type(obj).__name__} is not a dataclass instance"
    return asdict(obj)

# ===== TRAINING CONFIG ===== 
@dataclass
class PPOConfig:
    class_name: str = "PPO"
    clip_param: float = 0.2
    desired_kl: float = 0.01
    entropy_coef: float = 0.004
    gamma: float = 0.999 # orig 0.99
    lam: float = 0.99 # orig 0.95
    learning_rate: float = 0.0003
    max_grad_norm: float = 1.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    schedule: str = "adaptive"
    use_clipped_value_loss: bool = True
    value_loss_coef: float = 1.0

@dataclass
class GaussianDistributionConfig:
    class_name: str = "GaussianDistribution"
    init_std: float = 1.0
    std_type: str = "scalar"

@dataclass
class ActorConfig:
    class_name: str = "MLPModel"
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"
    distribution_cfg: GaussianDistributionConfig = field(default_factory=GaussianDistributionConfig)

@dataclass
class CriticConfig:
    class_name: str = "MLPModel"
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"

@dataclass
class ObsGroupsConfig:
    actor: list[str] = field(default_factory=lambda: ["policy"])
    critic: list[str] = field(default_factory=lambda: ["policy"])

@dataclass
class TrainConfig:
    algorithm: PPOConfig = field(default_factory=PPOConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    obs_groups: ObsGroupsConfig = field(default_factory=ObsGroupsConfig)
    num_steps_per_env: int = 1000 # orig 100
    save_interval: int = 100
    run_name: str = ""
    logger: str = "tensorboard"

# ====== TETHER CONFIG ======

@dataclass
class TetherConfig:
    num_tethers: int = 3
    stiffness: float =  5000.0
    damping: float = 100.0
    rest_length: float = 3.0
    diameter: float = 0.005
    slack_transition_width: float = 0.1
    activation_delta: float = max(slack_transition_width * rest_length, EPS)
    twist_stiffness: float = 5.0 # resist relative yaw winding of tethers
    twist_damping: float = 0.5

# ====== ENVIRONMENT CONFIG ======

@dataclass
class EnvConfig:
    num_envs: int = 1
    dt: float = 0.01
    max_sim_step_n: int = 2000
    episode_length_s: float = 30.0
    num_actions: int = 4
    drone_reset_pos: tuple[float, float, float] = (0.0, 0.0, 1.068)
    drone_reset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0) # wxyz
    payload_reset_pos: tuple[float, float, float] = (0.0, 0.0, 0.025)
    payload_reset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0) # wxyz
    air_density: float = 1.225
    terminate_if_roll_greater_than: float = 45 # payload body tilt (orig 90 -- too late to recover by then)
    terminate_if_pitch_greater_than: float = 45
    terminate_if_swing_angle_greater_than: float = 45 # tether angle from vertical (drone-payload line)
    terminate_if_x_greater_than: float = 10.0 # payload x
    terminate_if_y_greater_than: float = 10.0 # payload y
    terminate_if_z_greater_than: float = 10.0 # payload agl
    at_target_th: float = 0.5
    at_target_swing_vel_th: float = 0.3 # m/s, drone-payload relative xy velocity must also be low to count as "settled"
    resampling_time_s: float = 1.0
    simulate_action_latency: bool = True
    clip_actions: float = 1.0 # should be <= 1.0

    # Actuation: action = [thrust, wx, wy, wz]
    thrust_to_weight: float = 2.0   # thrust_max = thrust_to_weight * total_weight
    max_rate: float = 10.0           # rad/s, max commanded body rate magnitude
    rate_kp: float = 0.1
    rate_ki: float = 0.05
    rate_kd: float = 0.0001
    rate_integral_limit: float = 1.0

# ====== OBSERVATION CONFIG ======

@dataclass
class ObservationConfig:
    num_obs: int = 33
    scale_rel_pos: float = 1.0 / 3.0
    scale_lin_vel: float = 1.0 / 3.0
    scale_ang_vel: float = 1.0 / 3.14159
    scale_rel_swing: float = 1.0 / TetherConfig.rest_length # OBS: Could be a better solution to get the lenght here

# ====== COMMAND CONFIG =======

@dataclass
class CommandConfig:
    num_commands: int = 3
    pos_x_range: tuple[float, float, float] = (-3.0, 3.0)
    pos_y_range: tuple[float, float, float] = (-3.0, 3.0)
    pos_z_range: tuple[float, float, float] = (1.5, 3.0) # attempt to make hover at fixed z

# ======= REWARD CONFIG ======

@dataclass
class RewardConfig:
    sigma_target: float = 1.0
    max_close_rate: float = 2.0 # m/s, caps the rewarded rate of closing distance to target (orig: unbounded -> rewarded max-speed dashes)

    scale_target: float = 10.0
    scale_attitude: float = 5.0
    scale_action: float = -1e-4
    scale_tension: float = 1.0
    scale_crash: float = -25.0

    # Swing/motion damping (previously computed but never added into the reward sum)
    scale_swing_pos: float = 1.0 # penalize drone-payload lateral offset (pendulum displacement)
    scale_swing_vel: float = 1.0 # penalize drone-payload lateral relative velocity (pendulum rate)
    scale_swing_angle: float = 3.0 # penalize tether angle from vertical directly
    scale_smooth_accel: float = 0.05 # penalize payload jerk/acceleration
    scale_vel_damp: float = 1.0 # penalize residual payload velocity once close to target
