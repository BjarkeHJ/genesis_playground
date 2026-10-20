from dataclasses import dataclass, field
import math
from system_model.utils.qc_rate_control import RateControlParams, MotorPlantParams
from system_model.utils.qc_attitude_control import AttitudeControlParams

EPS = 1e-9

# ====== DOMAIN RAND ======
@dataclass(frozen=True)
class DomainRandConfig:
    enabled: bool = True

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

    # Success rate (logging only): episode reaches timeout without crash, mean payload error to the reference below this
    success_interval: int = 1200 # policy steps per success window
    success_pos_err: float = 0.3 # mean payload pos error over the episode that counts as success [m]

# ====== TETHER CONFIG ======
@dataclass(frozen=True)
class SystemConfig:
    # Tether
    num_tethers: int = 3
    stiffness: float = 1500.0
    damping: float = 200.0
    torsional_damping: float = 2e-4 # suspension yaw friction on the drone-payload relative yaw rate [N·m·s/rad]
    rest_length: float = 2.0
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


# ====== ROUTE SAMPLER CONFIG =======
@dataclass(frozen=True)
class RouteSamplerConfig:
    route_keywords: tuple[str, ...] = ("random", "survey", "loop", "zigzag", "stop")
    num_points_min: int = 5
    num_points_max: int = 20
    turn_max: float = math.radians(90.0)
    alt_min: float = 1.0
    alt_max: float = 3.0
    seg_length_min: float = 1.0
    seg_length_max: float = 10.0
    v_des_min: float = 1.0
    v_des_max: float = 2.0
    
    # probabilities
    p_straight: float = 0.3
    p_climb: float = 0.2
    
# ====== ROUTE CONFIG ======
@dataclass(frozen=True)
class RouteConfig:
    # Route -> payload reference. Only the mission spec and the system envelope live here; the generator's internal
    # shaping constants are in components/route.py and components/trajectory.py and are not meant to be tuned
    # Mission spec
    corridor: float = 0.5 # max deviation of a rounded corner from its waypoint [m]
    min_altitude: float = 1.0 # payload [m]

    # System envelope: limits on the payload reference after time-scaling (all durations stretched by one factor k).
    # Corner and ramp accelerations used for planning are fixed fractions of a_max
    a_max: float = 2.0 # [m/s²]
    j_max: float = 3.0 # [m/s³]

# ====== REFERENCE CONFIG ======
@dataclass(frozen=True)
class ReferenceConfig:
    # Online per-env reference on top of the bank (not part of the bank's cache key). Yaw is not optimized with the
    # position: a rate-limited follower turns the yaw reference towards the direction of the reference velocity,
    # averaged over a lookahead window so slow yaw turns start before the corner
    yaw_mode: str = "velocity" # "velocity": follow the (lookahead) velocity direction | "fixed": hold the initial heading
    yaw_tau: float = 0.75 # follower time constant [s]: larger follows corners more loosely
    yaw_rate_max: float = math.radians(30.0) # yaw-rate reference limit, below SystemConfig.max_yaw_rate (45°/s) to leave room for corrections
    yaw_min_speed: float = 0.3 # below this (lookahead) horizontal reference speed the yaw reference holds (stops, hover) [m/s]
    yaw_lookahead_s: float = 2.0 # window [0, H] of future reference velocities averaged for the heading target; 0: current velocity [s]
    yaw_lookahead_decay: float = 1.0 # weights exp(-t / decay) over the window: the near future dominates [s]

    # Observation and reset
    preview_s: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0) # preview points r(τ + τ_k) - payload, observed in the heading frame [s]
    p_mid_route_start: float = 0.0 # chance an episode starts at a uniform time along its route instead of the start
    init_yaw_err: float = math.radians(90.0) # initial payload yaw within this of the yaw reference

# ====== TRAJECTORY BANK CONFIG ======
@dataclass(frozen=True)
class TrajectoryBankConfig:
    # Routes sampled with RouteSamplerConfig and solved with RouteConfig once, cached under logs/route_bank
    size: int = 20_000
    max_segments: int = 128 # padded length; longer routes are redrawn
    keyword_weights: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2) # per RouteSamplerConfig.route_keywords
    seed: int = 0
    num_workers: int = 8 # processes for the build (not part of the cache key)

# ======= Observation Scales =======
@dataclass(frozen=True)
class ObservationScales:
    px: float = 1.0 / 3.0
    py: float = 1.0 / 3.0
    pz: float = 1.0 / 3.0
    vx: float = 1.0 / 6.0
    vy: float = 1.0 / 6.0
    vz: float = 1.0 / 6.0
    v_err: float = 1.0 / 3.0 # reference velocity - payload velocity
    preview_margin: float = 3.0 # preview point k is scaled by 1 / (v_des_max * τ_k + preview_margin): far points don't saturate [m]
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
    w_track: float = 1.0

    delta_vel_track: float = 0.2 # pseudo-Huber knee on |reference velocity - payload velocity| [m/s]
    w_vel_track: float = -0.5 # per m/s of velocity error (outside the knee). At the route end v_ref = 0: the old damping term
        
    delta_yaw: float = 0.2 # pseudo-Huber knee on payload yaw err: quadratic inside, linear outside [rad]
    w_yaw: float = -0.3 # per rad of payload yaw err (outside the knee)

    delta_yaw_rate_track: float = 0.2 # pseudo-Huber knee on payload yaw rate - yaw-rate reference [rad/s]
    w_yaw_rate_track: float = -0.3 # per rad/s of yaw-rate error (outside the knee)
    
    w_swing_energy: float = -1.0 # per rad of equivalent swing amplitude
    w_tilt: float = -1.0 # per (1 - cos(payload tilt))

    w_smooth_actions: tuple[float, float, float, float] = (-0.1, -0.1, -0.1, -0.1) # per Δa² [thrust, roll, pitch, yawrate]
    w_action_bound: float = -5.0 # per (pre-clip |a| - clip_actions)², summed over actions

    w_crash: float = -10.0

# ====== ENVIRONMENT CONFIG ======
@dataclass(frozen=True)
class EnvConfig:
    sys_cfg: SystemConfig = field(default_factory=SystemConfig)
    route_cfg: RouteConfig = field(default_factory=RouteConfig)
    sampler_cfg: RouteSamplerConfig = field(default_factory=RouteSamplerConfig)
    bank_cfg: TrajectoryBankConfig = field(default_factory=TrajectoryBankConfig)
    ref_cfg: ReferenceConfig = field(default_factory=ReferenceConfig)
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
    num_obs: int = field(init=False) # 33 + 3 per preview point, see PayloadControlEnv._compute_observations
    num_privileged_obs: int = 8 # critic only, DomainRandomizer.privileged_obs
    num_actions: int = 4

    # Actions
    simulate_action_latency: bool = True
    clip_actions: float = 1.0

    # Terminate conditions
    episode_length_s: float = 20.0
    terminate_if_payload_tilt_greater_than: float = math.pi / 4.0
    terminate_if_swingangle_greater_than: float = math.radians(60.0)
    terminate_if_x_greater_than: float = 3.0 # on pl pos err (bbox around the moving reference, heading frame)
    terminate_if_y_greater_than: float = 3.0
    terminate_if_z_greater_than: float = 3.0
    terminate_if_relyaw_greater_than: float = 0.75 * math.pi # below pi so the wrapped rel yaw never wraps before terminating
    terminate_if_payload_below_z: float = 0.2 # ground crash
    terminate_if_drone_tilt_greater_than: float = math.radians(70.0)

    # Post-init computations of dependent parameters
    def __post_init__(self):
        object.__setattr__(self, "dt", self.sim_dt * self.decimation)
        object.__setattr__(self, "num_obs", 33 + 3 * len(self.ref_cfg.preview_s))
        # Latency is applied by holding the previous setpoint for the first sim steps of a policy step, so it caps at one policy step
        if round(self.dr_cfg.latency_s[1] / self.sim_dt) > self.decimation:
            raise ValueError(f"latency_s max {self.dr_cfg.latency_s[1]} s exceeds one policy step ({self.dt} s)")



