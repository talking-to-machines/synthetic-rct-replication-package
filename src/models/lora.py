from dataclasses import dataclass, asdict


@dataclass
class LoRAConfig:
    """LoRA hyperparameters used for Together AI fine-tuning jobs."""

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_trainable_modules: str = "all-linear"

    def __post_init__(self):
        if self.lora_alpha is None:
            self.lora_alpha = 2 * self.lora_r

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Optimiser and schedule hyperparameters for Together AI fine-tuning jobs."""

    n_epochs: int = 6
    n_checkpoints: int = 1
    n_evals: int = 0
    batch_size: int = 8
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1
    weight_decay: float = 0
    train_on_inputs: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)
