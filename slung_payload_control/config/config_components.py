import math
from dataclasses import dataclass

# Component configs. Timing (sim_dt, decimation) is not repeated here: components take it from EnvConfig

# Flight Controller
@dataclass(frozen=True)
class FlightControllerConfig:
    n_rotors: int = 4 # must match the bound airframe
    clip_values: float = 1.0 # clip action values
    twr_max: float = 2.5 # thrust to weight ratio at full throttle (drone + payload)
    max_throttle: float = 1.0 # collective thrust setpoint cap [0, 1]
    max_tilt: float = math.radians(30.0) # roll/pitch setpoint bounds [rad]
    max_yaw_rate: float = math.radians(45.0) # yaw-rate setpoint range [rad/s]
    thrust_tilt_comp: bool = True # divide collective by cos(tilt_sp) so a_thrust = 0 approx holds altitude when tilted
    tilt_comp_max: float = math.radians(60.0) # cap on the compensated tilt

# Tether Model
@dataclass(frozen=True)
class TetherModelConfig:
    num_tethers: int = 3
    rest_length: float = 3.0
    stiffness: float = 1500.0
    damping: float = 200.0
    torsional_damping: float = 2e-4
    gate_tension: float = 0.3 # [N] 

# Domain Randomization
@dataclass(frozen=True)
class Uniform:
    lo: float
    hi: float
    nominal: float = 1.0
@dataclass(frozen=True)
class LogUniform(Uniform): ... # same but different name
@dataclass(frozen=True)
class DomainRandomizationConfig:
    enabled: bool = True

    # FlightController actuation
    thrust_scale: Uniform = Uniform(0.85, 1.15)
    latency_s: Uniform = Uniform(0.0, 0.020, nominal=0.0)

    # Tether
    rest_length_scale: Uniform = Uniform(0.85, 1.15)
    stiffness_scale: LogUniform = LogUniform(0.3, 3.0) 
    damping_scale: LogUniform = LogUniform(0.3, 3.0)
    torsional_damping_scale: LogUniform = LogUniform(0.25, 4.0)

    # Body masses
    payload_mass_scale: Uniform = Uniform(0.8, 1.2)
    drone_mass_scale: Uniform = Uniform(0.8, 1.2)

# Trajectory
@dataclass(frozen=True)
class TrajectoryConfig:
    # Route categories and their likelihood in training
    keywords: tuple[str, ...] = ("random", "survey", "loop", "zigzag", "stop")
    keyword_weights: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)

    # RouteSampler parameters
    route_points_min: int = 5
    route_points_max: int = 30
    route_seg_l_min: float = 1.0
    route_seg_l_max: float = 5.0
    route_alt_min: float = 5.0
    route_alt_max: float = 10.0
    route_vel_des_min: float = 2.0
    route_vel_des_max: float = 3.0
    route_max_climb_ang: float = math.radians(15.0)

    # Constraints
    corridor: float = 0.5 # max deviation from waypoints at a rounded corner
    a_max: float = 6.0 # prev 4.0
    j_max: float = 8.0 # prev 6.0

    # TrajectoryBank configuration
    bank_size: int = 20_000
    traj_max_segments: int = 128 # for padding; longer are re-drawn
    traj_max_duration: float = 25.0 # [s] is overwritten from EnvConfig's episode_length_s
    seed: int = 0
    num_workers: int = 8 # processes for the offline build