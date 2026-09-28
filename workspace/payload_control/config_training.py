from dataclasses import dataclass, field

@dataclass
class PPOConfig:
    class_name: str = "PPO"
    clip_param: float = 0.2
    desired_kl: float = 0.01
    entropy_coef: float = 0.004
    gamma: float = 0.99 
    lam: float = 0.95 
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
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
    distribution_cfg: GaussianDistributionConfig = field(default_factory=GaussianDistributionConfig)

@dataclass
class CriticConfig:
    class_name: str = "MLPModel"
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
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
    num_steps_per_env: int = 256
    save_interval: int = 100
    run_name: str = ""
    logger: str = "tensorboard"
    empirical_normalization: bool = True # does data-driven normalization of observation vector (currently no alternative)