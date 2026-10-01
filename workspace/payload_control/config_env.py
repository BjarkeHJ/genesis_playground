import os
from dataclasses import dataclass, field
import math
from system_model.utils.qc_rate_control import RateControlParams, MotorPlantParams
from system_model.utils.qc_attitude_control import AttitudeControlParams

EPS = 1e-9
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DRONE_PATH = os.path.join(REPO_ROOT, "system_model", "Starling2MaxURDF", "model", "starling2max.urdf")
PAYLOAD_PATH = os.path.join(REPO_ROOT, "system_model", "payload", "box", "box.urdf")

# ====== TETHER CONFIG ======
@dataclass(frozen=True)
class SystemConfig:
    # Tether
    num_tethers: int = 3
    stiffness: float = 1500.0
    damping: float = 200.0
    rest_length: float = 2.0
    diameter: float = 0.005
    slack_transition_width: float = 0.1
    activation_delta: float = field(init=False) # computed in post-init

    # Payload
    payload_init_z: float = 5.0
    payload_reset_pos: tuple[float, float, float] = field(init=False) # computed in post-init
    payload_reset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0) #wxyz

    # Drone
    drone_reset_pos: tuple[float, float, float] = field(init=False) # computed in post-init
    drone_reset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0) #wxyz
    twr_max: float = 2.5
    max_throttle: float = 1.0
    propellers_link_name: tuple[str, ...] = ("prop0_link", "prop1_link", "prop2_link", "prop3_link")
    propellers_spin: tuple[int, ...] = (-1, 1, -1, 1)
    rotor_ct: float = 6.5

    max_tilt: float = math.radians(30.0) # roll/pitch setpoint bounds [rad]
    max_yaw_rate: float = math.radians(90.0) # yaw-rate setpoint range [rad/s]
    thrust_tilt_comp: bool = True # divide collective by cos(tilt_sp) sp a_thrust = 0 will approx hold altitude when tilted
    tilt_comp_max: float = math.radians(60.0) # cap on comp

    rate_control_params: RateControlParams = field(default_factory=RateControlParams)
    motor_plant_params: MotorPlantParams = field(default_factory=MotorPlantParams)
    attitude_control_params: AttitudeControlParams = field(default_factory=AttitudeControlParams)

    # Post-init computations of dependent parameters
    def __post_init__(self):
        object.__setattr__(self, "activation_delta", max(self.slack_transition_width * self.rest_length, EPS))
        object.__setattr__(self, "payload_reset_pos", (0.0, 0.0, self.payload_init_z)) 
        object.__setattr__(self, "drone_reset_pos", (0.0, 0.0, self.payload_init_z + self.rest_length)) # start approx at suspension

# ====== COMMAND CONFIG =======
@dataclass(frozen=True)
class CommandConfig:
    num_waypoints: int = 3
    num_commands: int = 3

    pos_x_range: tuple[float, float] = (-3.0, 3.0)
    pos_y_range: tuple[float, float] = (-3.0, 3.0)
    pos_z_range: tuple[float, float] = (1.5, 3.0)

# ======= Observation Scales =======
@dataclass(frozen=True)
class ObservationScales:
    px: float = 1.0 / 3.0
    py: float = 1.0 / 3.0
    pz: float = 1.0 / 3.0
    vx: float = 1.0 / 6.0 # v_max maps to 0.5 so overspeed stays visible
    vy: float = 1.0 / 6.0
    vz: float = 1.0 / 6.0
    rp: float = 1.0 / (math.pi / 4)
    rpr: float = 1.0 / math.pi # measure?
    sa: float = 1.0 / (math.pi / 4)
    sar: float = 1.0 / 2.0 # measure?
    ryaw: float = 1.0 / math.pi
    drp: float = 1.0 / math.radians(45.0) # drone roll/pitch

# ======= REWARD CONFIG ======
@dataclass(frozen=True)
class RewardConfig:
    sigma_track_rough: float = 3.0
    sigma_track_fine: float = 0.3
    w_track_rough: float = 1.0
    w_track_fine: float = 1.5

    v_max: float = 3.0 # payload speed limit [m/s]
    w_vmax: float = -1.0 # per (m/s)² above v_max

    w_swing_energy: float = -1.0 # per rad of equivalent swing amplitude
    w_smooth_actions: float = -0.1
    w_crash: float = -10.0

# ====== ENVIRONMENT CONFIG ======
@dataclass(frozen=True)
class EnvConfig:
    sys_cfg: SystemConfig = field(default_factory=SystemConfig)
    cmd_cfg: CommandConfig = field(default_factory=CommandConfig)
    rew_cfg: RewardConfig = field(default_factory=RewardConfig)
    obs_scales: ObservationScales = field(default_factory=ObservationScales)

    # Physics engine
    num_envs: int = 1

    # Timing: policy runs every `decimation` sim steps; rate controller + tether model run every sim step
    sim_dt: float = 0.0025 # physics / rate controller step [s]
    sim_substeps: int = 1 # internal physics substeps per sim step
    decimation: int = 8 # sim steps per policy step
    dt: float = field(init=False) # policy step [s]

    # Policy input and output layer dim
    num_obs: int = 21
    num_actions: int = 4

    # Actions
    simulate_action_latency: bool = True
    clip_actions: float = 1.0

    # Terminate conditions
    episode_length_s: float = 10.0
    terminate_if_rollpitch_greater_than: float = math.pi / 4.0
    terminate_if_swingangle_greater_than: float = math.radians(60.0)
    terminate_if_x_greater_than: float = 7.0 # on pl pos err (bbox around target)
    terminate_if_y_greater_than: float = 7.0 
    terminate_if_z_greater_than: float = 7.0 
    terminate_if_relyaw_greater_than: float = math.pi
    terminate_if_payload_below_z: float = 0.2 # on ground
    terminate_if_drone_tilt_greater_than: float = math.radians(70.0)

    at_target_th: float = 0.5

    # Post-init computations of dependent parameters
    def __post_init__(self):
        object.__setattr__(self, "dt", self.sim_dt * self.decimation)



