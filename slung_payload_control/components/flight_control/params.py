from dataclasses import dataclass
import math

FLT_EPS = 1.1920929e-07 # floating point epsilon
ROLL, PITCH, YAW, THRUST_Z = 0, 1, 2, 3 # axis indices of the control vector

@dataclass(frozen=True)
class RateControlParams:
    # PX4 Parameters (defaults)
    MC_ROLLRATE_P: float = 0.15
    MC_ROLLRATE_I: float = 0.2
    MC_ROLLRATE_D: float = 0.003
    MC_ROLLRATE_FF: float = 0.0
    MC_ROLLRATE_K: float = 1.0
    MC_RR_INT_LIM: float = 0.30
    MC_PITCHRATE_P: float = 0.15
    MC_PITCHRATE_I: float = 0.2
    MC_PITCHRATE_D: float = 0.003
    MC_PITCHRATE_FF: float = 0.0
    MC_PITCHRATE_K: float = 1.0
    MC_PR_INT_LIM: float = 0.30
    MC_YAWRATE_P: float = 0.2
    MC_YAWRATE_I: float = 0.1
    MC_YAWRATE_D: float = 0.0
    MC_YAWRATE_FF: float = 0.0
    MC_YAWRATE_K: float = 1.0
    MC_YR_INT_LIM: float = 0.30

    IMU_GYRO_CUTOFF: float = 40.0 # [Hz] 2nd-order Butterworth on the rate, 0 disables
    IMU_DGYRO_CUTOFF: float = 30.0 # [HZ] 1st-order lowpass on D-term input, 0 disables

    MC_AIRMODE: int = 0 # 0 diabled, 1 roll/pitch, 2 roll/pitch/yaw
    CA_R_SLEW: float = 0.0 

    THR_MDL_FAC: float = 1.0 # orig 0.0

@dataclass(frozen=True)
class MotorPlantParams:
    # ESC + motor model (From hardware)
    rpm_min_frac: float = 0.0 # rpm at cmd = 0.0
    tau_up: float = 0.0125 # [s] spin-up time constant
    tau_down: float = 0.025 # [s] spin-down time constant

@dataclass(frozen=True)
class AttitudeControlParams:
    # PX4 Parameters (defaults)
    MC_ROLL_P: float = 6.5
    MC_PITCH_P: float = 6.5
    MC_YAW_P: float = 2.8
    MC_YAW_WEIGHT: float = 0.4 # yaw priority relative to roll/pitch in the reduced attitude mix

    MC_ROLLRATE_MAX: float = 220.0 # [deg/s]
    MC_PITCHRATE_MAX: float = 220.0 # [deg/s]
    MC_YAWRATE_MAX: float = 200.0 # [deg/s]

    yaw_hold: bool = True # True: integrate yaw setpoint from yaw rate (heading hold), False: pure yaw-rate
    yaw_err_max: float = math.radians(45.0) # [rad] cap on |yaw_sp - yaw| so the integrated setpoint cannot run away
