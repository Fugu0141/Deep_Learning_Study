"""Deep Learning Studio: learn and inspect a GPT-style language model locally."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ModelConfig, TrainingConfig
    from .model import TinyGPT
    from .tokenizer import BPETokenizer

__all__ = ["BPETokenizer", "ModelConfig", "TinyGPT", "TrainingConfig"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "BPETokenizer":
        from .tokenizer import BPETokenizer

        return BPETokenizer
    if name in {"ModelConfig", "TrainingConfig"}:
        from . import config

        return getattr(config, name)
    if name == "TinyGPT":
        from .model import TinyGPT

        return TinyGPT
    raise AttributeError(name)
