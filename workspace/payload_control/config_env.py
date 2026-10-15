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

# ====== DOMAIN RAND ======
@dataclass(frozen=True)
class DomainRandConfig:
    enabled: bool = True
    curriculum: bool = False # ramp one level 0 -> 1 that scales every enabled group's range (off: full range from the start)
    groups: tuple[str, ...] = ("init", "yaw", "tether", "bodies", "actuation") # enabled groups; left out = nominal ("yaw" left out = full yaw_ref range)

    # Tether
    rest_length_scale: tuple[float, float] = (0.8, 1.2) # effective pendulum length != tether rest length
    stiffness_scale: tuple[float, float] = (0.3, 3.0) # log-uniform
    damping_scale: tuple[float, float] = (0.3, 3.0) # log-uniform
    torsional_damping_scale: tuple[float, float] = (0.25, 4.0) # log-uniform

    # Bodies
    payload_mass_scale: tuple[float, float] = (0.8, 1.15)
    drone_mass_scale: tuple[float, float] = (0.9, 1.1)

    # Actuation
    thrust_scale: tuple[float, float] = (0.85, 1.15) # KF / battery voltage
    latency_s: tuple[float, float] = (0.0, 0.020) # [s] setpoint latency, rounded to whole sim steps (5 ms at sim_dt=5 ms)

    # Initial state
    init_swing_angle: float = math.radians(20.0)
    init_swing_rate: float = 0.5
    init_drone_vel: float = 0.5
    init_rel_yaw: float = math.radians(10.0)

    # Curriculum (if curriculum = True). After every level change one episode passes before success is measured again
    curriculum_interval: int = 1200 # policy steps per success window
    curriculum_step: float = 0.2 # level increase when success > curriculum_up (5 steps to full)
    curriculum_step_down: float = 0.2 # level decrease when success < curriculum_down
    curriculum_up: float = 0.8 # success rate to advance
    curriculum_down: float = 0.5 # success rate to back off

    curriculum_window_s: float = 3.0 # 150 policy steps
    curriculum_pos_err: float = 0.25 # final payload pos error that counts as success (trained policy settles to ~0.05 m)

# ====== TETHER CONFIG ======
@dataclass(frozen=True)
class SystemConfig:
    # Tether
    num_tethers: int = 3
    stiffness: float = 1500.0
    damping: float = 200.0
    torsional_damping: float = 2e-4 # suspension yaw friction on the drone-payload relative yaw rate [N·m·s/rad]
    rest_length: float = 5.0
    activation_delta: float = 0.2

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
    max_yaw_rate: float = math.radians(45.0) # yaw-rate setpoint range [rad/s]
    thrust_tilt_comp: bool = True # divide collective by cos(tilt_sp) sp a_thrust = 0 will approx hold altitude when tilted
    tilt_comp_max: float = math.radians(60.0) # cap on comp

    rate_control_params: RateControlParams = field(default_factory=RateControlParams)
    motor_plant_params: MotorPlantParams = field(default_factory=MotorPlantParams)
    attitude_control_params: AttitudeControlParams = field(default_factory=AttitudeControlParams)

    # Post-init computations of dependent parameters
    def __post_init__(self):
        object.__setattr__(self, "payload_reset_pos", (0.0, 0.0, self.payload_init_z)) 
        object.__setattr__(self, "drone_reset_pos", (0.0, 0.0, self.payload_init_z + self.rest_length)) # start approx at suspension


# ====== COMMAND CONFIG =======
@dataclass(frozen=True)
class CommandConfig:
    num_waypoints: int = 3
    num_commands: int = 4 # [x, y, z, yaw_ref]

    pos_x_range: float = 3.0
    pos_y_range: float = 3.0
    pos_z_range: float = 2.0
    yaw_range: float = math.radians(90.0) # yaw_ref around initial payload heading (full range, scaled by the curriculum level when "yaw" is in DomainRandConfig.groups)

# ======= Observation Scales =======
@dataclass(frozen=True)
class ObservationScales:
    px: float = 1.0 / 3.0
    py: float = 1.0 / 3.0
    pz: float = 1.0 / 3.0
    vx: float = 1.0 / 6.0 # v_max maps to 0.5 so overspeed stays visible
    vy: float = 1.0 / 6.0
    vz: float = 1.0 / 6.0
    pl_body_rates: float = 1.0 / math.pi # size?
    pl_swing_angles: float = 1.0 / math.radians(60.0) # = 1 / terminate_if_swingangle_greater_than
    pl_swing_angles_rates: float = 1.0 / 1.0 # measure?
    rel_yaw: float = 1.0 / (0.75 * math.pi) # = 1 / terminate_if_relyaw_greater_than
    drone_body_rates: float = 1.0 / math.pi

# ======= REWARD CONFIG ======
@dataclass(frozen=True)
class RewardConfig:
    sigma_track_rough: float = 3.0
    sigma_track_fine: float = 0.3
    w_track_rough: float = 1.0
    w_track_fine: float = 1.5

    v_max: float = 3.0 # payload speed limit [m/s]
    w_vmax: float = -1.0 # per (m/s)² above v_max

    sigma_damping: float = 0.5 # gate on payload speed penalty near the target [m]
    w_damping: float = -0.75 # per m/s of payload speed at the target
    w_swing_energy: float = -1.0 # per rad of equivalent swing amplitude
    
    delta_yaw: float = 0.1 # pseudo-Huber knee on payload yaw err: quadratic inside, linear outside [rad]
    w_yaw: float = -0.3 # per rad of payload yaw err (outside the knee)
    sigma_yaw_damping: float = 0.3 # gate on payload yaw rate penalty near the yaw target [rad]
    delta_yaw_damping: float = 0.2 # pseudo-Huber knee on payload yaw rate [rad/s]
    w_yaw_damping: float = -0.3 # per rad/s of payload yaw rate at the yaw target (outside the knee)

    w_tilt: float = -1.0 # per (1 - cos(payload tilt))
    w_smooth_actions: tuple[float, float, float, float] = (-0.1, -0.1, -0.1, -0.1) # per Δa² [thrust, roll, pitch, yawrate]
    w_action_bound: float = -1.0 # per (pre-clip |a| - clip_actions)², summed over actions
    w_crash: float = -10.0

# ====== ENVIRONMENT CONFIG ======
@dataclass(frozen=True)
class EnvConfig:
    sys_cfg: SystemConfig = field(default_factory=SystemConfig)
    cmd_cfg: CommandConfig = field(default_factory=CommandConfig)
    rew_cfg: RewardConfig = field(default_factory=RewardConfig)
    dr_cfg: DomainRandConfig = field(default_factory=DomainRandConfig)
    obs_scales: ObservationScales = field(default_factory=ObservationScales)

    # Physics engine
    num_envs: int = 1

    # Timing: policy runs every `decimation` sim steps; rate controller + tether model run every sim step
    sim_dt: float = 0.005 # physics / rate controller step [s] (200 Hz; real PX4 rate loop runs at IMU_GYRO_RATEMAX)
    sim_substeps: int = 1 # internal physics substeps per sim step
    decimation: int = 4 # sim steps per policy step
    dt: float = field(init=False) # policy step [s]
    torch_compile: bool = True # fuse the per-sim-step torch code (controllers, tethers); costs compile time at startup

    # Policy input and output layer dim
    num_obs: int = 29
    num_actions: int = 4

    # Actions
    simulate_action_latency: bool = True
    clip_actions: float = 1.0

    # Terminate conditions
    episode_length_s: float = 20.0
    terminate_if_payload_tilt_greater_than: float = math.pi / 4.0
    terminate_if_swingangle_greater_than: float = math.radians(60.0)
    terminate_if_x_greater_than: float = 7.0 # on pl pos err (bbox around target, heading frame)
    terminate_if_y_greater_than: float = 7.0
    terminate_if_z_greater_than: float = 7.0
    terminate_if_relyaw_greater_than: float = 0.75 * math.pi # below pi so the wrapped rel yaw never wraps before terminating
    terminate_if_payload_below_z: float = 0.2 # ground crash
    terminate_if_drone_tilt_greater_than: float = math.radians(70.0)

    # Post-init computations of dependent parameters
    def __post_init__(self):
        object.__setattr__(self, "dt", self.sim_dt * self.decimation)
        # Latency is applied by holding the previous setpoint for the first sim steps of a policy step, so it caps at one policy step
        if round(self.dr_cfg.latency_s[1] / self.sim_dt) > self.decimation:
            raise ValueError(f"latency_s max {self.dr_cfg.latency_s[1]} s exceeds one policy step ({self.dt} s)")



