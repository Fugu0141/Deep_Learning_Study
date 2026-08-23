from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from .config import ModelConfig, TrainingConfig
from .model import TinyGPT
from .tokenizer import BPETokenizer


@dataclass(slots=True)
class TrainingMetrics:
    step: int
    max_steps: int
    train_loss: float
    validation_loss: float | None
    perplexity: float | None
    learning_rate: float
    gradient_norm: float
    tokens_per_second: float
    elapsed_seconds: float
    sample_prediction: str = ""
    attention: list[list[float]] | None = None
    attention_tokens: list[str] | None = None


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is not available")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was selected but is not available")
    return torch.device(requested)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        memory_gib = torch.cuda.get_device_properties(device).total_memory / 1024**3
        return f"CUDA: {name} ({memory_gib:.1f} GiB)"
    if device.type == "mps":
        return "Apple Metal (MPS)"
    return f"CPU ({torch.get_num_threads()} threads)"


class LanguageModelTrainer:
    def __init__(
        self,
        model: TinyGPT,
        tokenizer: BPETokenizer,
        corpus: str,
        config: TrainingConfig,
    ) -> None:
        config.validate()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = select_device(config.device)
        token_ids = tokenizer.encode(corpus, add_bos=True, add_eos=True)
        minimum = model.config.context_length + 2
        if len(token_ids) < minimum * 2:
            raise ValueError(
                f"学習データが短すぎます。現在 {len(token_ids)} tokens、"
                f"この設定では最低 {minimum * 2} tokens 必要です。"
            )
        data = torch.tensor(token_ids, dtype=torch.long)
        split_index = int(len(data) * config.train_split)
        split_index = max(minimum, min(split_index, len(data) - minimum))
        self.train_data = data[:split_index]
        self.validation_data = data[split_index:]
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.stop_event = threading.Event()
        self.current_step = 0
        self.history: list[TrainingMetrics] = []

    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay: list[Tensor] = []
        no_decay: list[Tensor] = []
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.dim() >= 2 else no_decay).append(parameter)
        groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_available = (
            self.device.type == "cuda"
            and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        )
        return torch.optim.AdamW(
            groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            fused=fused_available,
        )

    def request_stop(self) -> None:
        self.stop_event.set()

    def _batch(self, data: Tensor) -> tuple[Tensor, Tensor]:
        context = self.model.config.context_length
        starts = torch.randint(0, len(data) - context - 1, (self.config.batch_size,))
        inputs = torch.stack([data[index : index + context] for index in starts])
        targets = torch.stack([data[index + 1 : index + context + 1] for index in starts])
        return inputs.to(self.device), targets.to(self.device)

    def _learning_rate(self, step: int) -> float:
        if self.config.warmup_steps and step < self.config.warmup_steps:
            return self.config.learning_rate * (step + 1) / self.config.warmup_steps
        decay_steps = max(1, self.config.max_steps - self.config.warmup_steps)
        progress = min(1.0, (step - self.config.warmup_steps) / decay_steps)
        minimum = self.config.learning_rate * 0.1
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum + cosine * (self.config.learning_rate - minimum)

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float, list[list[float]], list[str], str]:
        was_training = self.model.training
        self.model.eval()
        losses: list[float] = []
        last_inputs: Tensor | None = None
        last_output = None
        for batch_index in range(self.config.eval_batches):
            inputs, targets = self._batch(self.validation_data)
            capture = batch_index == self.config.eval_batches - 1
            output = self.model(inputs, targets, capture_attention=capture)
            assert output.loss is not None
            losses.append(float(output.loss.detach().cpu()))
            last_inputs, last_output = inputs, output

        attention_matrix: list[list[float]] = []
        attention_tokens: list[str] = []
        prediction = ""
        if last_inputs is not None and last_output is not None:
            display_length = min(24, last_inputs.shape[1])
            selected = last_inputs[0, -display_length:].detach().cpu().tolist()
            attention_tokens = [self.tokenizer.token_label(token_id) for token_id in selected]
            if last_output.attentions:
                weights = last_output.attentions[-1][0].mean(dim=0)
                attention_matrix = (
                    weights[-display_length:, -display_length:].detach().float().cpu().tolist()
                )
            next_id = int(last_output.logits[0, -1].argmax().detach().cpu())
            prediction = self.tokenizer.token_label(next_id)
        self.model.train(was_training)
        mean_loss = sum(losses) / len(losses)
        return (
            mean_loss,
            math.exp(min(20.0, mean_loss)),
            attention_matrix,
            attention_tokens,
            prediction,
        )

    def train(
        self,
        on_metrics: Callable[[TrainingMetrics], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> list[TrainingMetrics]:
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self.stop_event.clear()
        self.model.train()
        mixed_precision = self.config.use_mixed_precision and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
        started = time.perf_counter()
        previous_time = started
        if on_status:
            on_status(f"学習開始: {describe_device(self.device)}")

        for step in range(self.current_step, self.config.max_steps):
            if self.stop_event.is_set():
                if on_status:
                    on_status("停止要求を受け取り、安全に学習を停止しました")
                break
            learning_rate = self._learning_rate(step)
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            inputs, targets = self._batch(self.train_data)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=mixed_precision,
            ):
                output = self.model(inputs, targets)
                assert output.loss is not None
                loss = output.loss
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            scaler.step(self.optimizer)
            scaler.update()
            self.current_step = step + 1

            should_report = step == 0 or self.current_step % self.config.eval_interval == 0
            if should_report or self.current_step == self.config.max_steps:
                validation_loss, perplexity, attention, tokens, prediction = self.evaluate()
                now = time.perf_counter()
                steps_in_window = 1 if len(self.history) == 0 else self.config.eval_interval
                tokens_per_second = (
                    steps_in_window
                    * self.config.batch_size
                    * self.model.config.context_length
                    / max(now - previous_time, 1e-9)
                )
                previous_time = now
                metrics = TrainingMetrics(
                    step=self.current_step,
                    max_steps=self.config.max_steps,
                    train_loss=float(loss.detach().cpu()),
                    validation_loss=validation_loss,
                    perplexity=perplexity,
                    learning_rate=learning_rate,
                    gradient_norm=float(gradient_norm.detach().cpu()),
                    tokens_per_second=tokens_per_second,
                    elapsed_seconds=now - started,
                    sample_prediction=prediction,
                    attention=attention,
                    attention_tokens=tokens,
                )
                self.history.append(metrics)
                if on_metrics:
                    on_metrics(metrics)

        if on_status and not self.stop_event.is_set():
            on_status("学習が完了しました")
        return self.history

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_config": self.model.config.to_dict(),
                "training_config": self.config.to_dict(),
                "tokenizer": self.tokenizer.to_dict(),
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": self.current_step,
                "history": [asdict(item) for item in self.history],
            },
            path,
        )


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "auto",
) -> tuple[TinyGPT, BPETokenizer, dict]:
    target_device = select_device(device)
    checkpoint = torch.load(Path(path), map_location=target_device, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    tokenizer = BPETokenizer.from_dict(checkpoint["tokenizer"])
    model = TinyGPT(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(target_device)
    model.eval()
    return model, tokenizer, checkpoint
