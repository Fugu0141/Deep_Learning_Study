from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


@dataclass(slots=True)
class ModelOutput:
    logits: Tensor
    loss: Tensor | None
    attentions: list[Tensor] | None


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=config.bias)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(config.context_length, config.context_length, dtype=torch.bool)
            ).view(1, 1, config.context_length, config.context_length),
            persistent=False,
        )

    def forward(
        self, inputs: Tensor, capture_attention: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        batch_size, sequence_length, embedding_dim = inputs.shape
        query, key, value = self.qkv(inputs).split(embedding_dim, dim=2)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        dropped_weights = self.attention_dropout(weights)
        attended = dropped_weights @ value
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        )
        output = self.residual_dropout(self.projection(attended))
        return output, weights if capture_attention else None


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.embedding_dim, bias=config.bias)
        self.feed_forward = FeedForward(config)

    def forward(
        self, inputs: Tensor, capture_attention: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        attention_output, weights = self.attention(
            self.attention_norm(inputs), capture_attention=capture_attention
        )
        hidden = inputs + attention_output
        hidden = hidden + self.feed_forward(self.ffn_norm(hidden))
        return hidden, weights


class TinyGPT(nn.Module):
    """Decoder-only Transformer language model trained with next-token prediction."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.embedding_dim, bias=config.bias)
        self.language_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.language_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)
        self._scale_residual_projections()

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        scale = 0.02 / math.sqrt(2 * self.config.num_layers)
        for name, parameter in self.named_parameters():
            if name.endswith(("projection.weight", "network.2.weight")):
                nn.init.normal_(parameter, mean=0.0, std=scale)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
        *,
        capture_attention: bool = False,
    ) -> ModelOutput:
        _, sequence_length = token_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length "
                f"{self.config.context_length}"
            )
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.embedding_dropout(
            self.token_embedding(token_ids) + self.position_embedding(positions)
        )
        attentions: list[Tensor] | None = [] if capture_attention else None
        for block in self.blocks:
            hidden, weights = block(hidden, capture_attention=capture_attention)
            if attentions is not None and weights is not None:
                attentions.append(weights)
        logits = self.language_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return ModelOutput(logits=logits, loss=loss, attentions=attentions)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.8,
        top_k: int | None = 40,
        repetition_penalty: float = 1.0,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
        on_token: Callable[[int, Tensor, Tensor | None, Tensor], None] | None = None,
    ) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be at least 1.0")
        was_training = self.training
        self.eval()
        generated = prompt_ids
        for step in range(max_new_tokens):
            context = generated[:, -self.config.context_length :]
            output = self(context, capture_attention=on_token is not None)
            logits = output.logits[:, -1, :].clone()
            if repetition_penalty != 1.0:
                for token_id in torch.unique(generated):
                    token_logits = logits[:, token_id]
                    logits[:, token_id] = torch.where(
                        token_logits < 0,
                        token_logits * repetition_penalty,
                        token_logits / repetition_penalty,
                    )
            logits /= temperature
            if top_k is not None and top_k > 0:
                threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            attention = None
            if output.attentions:
                attention = output.attentions[-1][0].detach().float().cpu()
            generated = torch.cat((generated, next_token), dim=1)
            if on_token is not None:
                on_token(
                    step,
                    next_token[0, 0].detach().cpu(),
                    attention,
                    probabilities[0].detach().float().cpu(),
                )
            if eos_id is not None and torch.all(next_token == eos_id):
                break
        self.train(was_training)
        return generated

    def parameter_count(self, *, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)
