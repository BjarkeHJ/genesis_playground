from dataclasses import dataclass

@dataclass
class EnvConfig:
    dt: float = 0.01
    max_sim_step_n: int = 2000
    air_density: float = 1.225
    num_obs: int = 0
    num_commands: int = 0
    num_actions: int = 0
    num_tethers: int = 3

@dataclass
class CableConfig:
    stiffness: float =  5000.0
    damping: float = 100.0
    rest_length: float = 5.0
    diameter: float = 0.005
    slack_transition_width: float = 0.01