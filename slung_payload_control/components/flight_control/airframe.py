from dataclasses import dataclass

@dataclass(frozen=True)
class Rotor:
    # One rotor position relative to CoM, body FRD
    px: float
    py: float
    km: float # moment ratio, > 0 for CCW rotor
    ct: float = 6.5 # PX4 default - drops out in most computations. Only matters for weak-axis cutoff in mixer

@dataclass(frozen=True)
class Airframe:
    # Everything the flight controller needs from the simulated vehicle (see utils.genesis_helpers.airframe_from_genesis)
    rotors: tuple[Rotor, ...]
    rpm_max: float # rpm at full actuator setpoint
