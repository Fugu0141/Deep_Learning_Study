from __future__ import annotations

import traceback
from dataclasses import asdict

import torch
from PySide6.QtCore import QObject, Signal, Slot

from ..model import TinyGPT
from ..tokenizer import BPETokenizer
from ..training import LanguageModelTrainer


class TrainingWorker(QObject):
    metrics = Signal(dict)
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, trainer: LanguageModelTrainer) -> None:
        super().__init__()
        self.trainer = trainer

    @Slot()
    def run(self) -> None:
        try:
            self.trainer.train(
                on_metrics=lambda item: self.metrics.emit(asdict(item)),
                on_status=self.status.emit,
            )
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self.trainer.request_stop()


class GenerationWorker(QObject):
    token = Signal(dict)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        model: TinyGPT,
        tokenizer: BPETokenizer,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        repetition_penalty: float,
        auto_stop: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.auto_stop = auto_stop

    @Slot()
    def run(self) -> None:
        try:
            device = next(self.model.parameters()).device
            ids = self.tokenizer.encode(self.prompt, add_bos=not self.prompt)
            if not ids:
                ids = [self.tokenizer.bos_id]
            prompt_tensor = torch.tensor([ids], dtype=torch.long, device=device)

            def on_token(step, token_id, attention, probabilities) -> None:
                k = min(8, probabilities.numel())
                values, indices = torch.topk(probabilities, k)
                labels = [self.tokenizer.token_label(int(index)) for index in indices]
                matrix = []
                tokens = []
                if attention is not None:
                    averaged = attention.mean(dim=0)
                    display_length = min(24, averaged.shape[0])
                    matrix = averaged[-display_length:, -display_length:].tolist()
                    context_ids = ids + generated_ids
                    tokens = [
                        self.tokenizer.token_label(value) for value in context_ids[-display_length:]
                    ]
                generated_ids.append(int(token_id))
                self.token.emit(
                    {
                        "step": step,
                        "token_id": int(token_id),
                        "text": self.tokenizer.decode([int(token_id)]),
                        "candidates": list(zip(labels, values.tolist())),
                        "attention": matrix,
                        "attention_tokens": tokens,
                    }
                )

            generated_ids: list[int] = []

            def should_stop(_all_tokens) -> bool:
                if not self.auto_stop or len(generated_ids) < 8:
                    return False
                generated_text = self.tokenizer.decode(generated_ids)
                return generated_text.endswith(("。", "！", "？", "!", "?", "\n"))

            output = self.model.generate(
                prompt_tensor,
                self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                eos_id=self.tokenizer.eos_id if self.auto_stop else None,
                on_token=on_token,
                should_stop=should_stop if self.auto_stop else None,
            )
            self.finished.emit(self.tokenizer.decode(output[0].detach().cpu().tolist()))
        except Exception:
            self.error.emit(traceback.format_exc())
