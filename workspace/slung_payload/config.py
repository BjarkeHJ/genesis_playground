from dataclasses import dataclass, asdict, is_dataclass

_EPS = 1e-9

def dataclass_to_dict(obj):
    assert is_dataclass(obj) and not isinstance(obj, type), f"{type(obj).__name__} is not a dataclass instance"
    return asdict(obj)

@dataclass
class EnvConfig:
    dt: float = 0.01
    max_sim_step_n: int = 2000
    num_actions: int = 4
    reset_pos: tuple[float, float, float] = (0.0, 0.0, 0.068)
    reset_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0) # wxyz
    air_density: float = 1.225

@dataclass
class TetherConfig:
    num_tethers: int = 3
    stiffness: float =  2500.0
    damping: float = 100.0
    rest_length: float = 5.0
    diameter: float = 0.005
    slack_transition_width: float = 0.01
    activation_delta: float = max(slack_transition_width * rest_length, _EPS)

@dataclass
class TrainingConfig:
    pass

@dataclass 
class RewardConfig:
    pass

@dataclass
class CommandConfig:
    num_commands: int = 3
    pos_x_range: tuple[float, float, float] = (-2.0, 2.0)
    pos_y_range: tuple[float, float, float] = (-2.0, 2.0)
    pos_z_range: tuple[float, float, float] = (10.0, 10.0) # attempt to make hover at fixed z

@dataclass
class ObservationConfig:
    num_obs: int = 3