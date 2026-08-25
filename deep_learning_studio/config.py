from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    embedding_dim: int = 192
    num_heads: int = 6
    num_layers: int = 6
    dropout: float = 0.1
    bias: bool = True

    def validate(self) -> None:
        if self.vocab_size < 4:
            raise ValueError("vocab_size must be at least 4")
        if self.context_length < 8:
            raise ValueError("context_length must be at least 8")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("num_layers and num_heads must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingConfig:
    max_steps: int = 1000
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 50
    eval_interval: int = 25
    eval_batches: int = 10
    gradient_clip: float = 1.0
    train_split: float = 0.9
    seed: int = 42
    device: str = "auto"
    use_mixed_precision: bool = True

    def validate(self) -> None:
        if self.max_steps < 1 or self.batch_size < 1:
            raise ValueError("max_steps and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.5 <= self.train_split < 1.0:
            raise ValueError("train_split must be in [0.5, 1.0)")
        if self.eval_interval < 1 or self.eval_batches < 1:
            raise ValueError("evaluation settings must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
