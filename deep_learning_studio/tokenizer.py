from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any


@dataclass(frozen=True, slots=True)
class Merge:
    left: str
    right: str
    merged: str


class BPETokenizer:
    """Small educational BPE tokenizer trained only from the user's corpus.

    The implementation starts from Unicode characters and repeatedly merges the
    most frequent adjacent pair. It is intentionally transparent rather than
    delegating tokenization to a pretrained external package.
    """

    PAD = "<|pad|>"
    UNK = "<|unk|>"
    BOS = "<|bos|>"
    EOS = "<|eos|>"
    SPECIAL_TOKENS = (PAD, UNK, BOS, EOS)

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        self.merges: list[Merge] = []
        self.base_characters: set[str] = set()

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    def train(
        self,
        text: str,
        target_vocab_size: int = 512,
        min_pair_frequency: int = 2,
    ) -> list[dict[str, Any]]:
        if not text:
            raise ValueError("Tokenizer training text must not be empty")
        if target_vocab_size < len(self.SPECIAL_TOKENS) + 1:
            raise ValueError("target_vocab_size is too small")

        self.base_characters = set(text)
        # One exact sequence preserves leading, repeated and trailing newlines.
        sequences = [list(text)]

        self.merges = []
        vocabulary = set(self.base_characters)
        history: list[dict[str, Any]] = []

        while len(vocabulary) + len(self.SPECIAL_TOKENS) < target_vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for sequence in sequences:
                pair_counts.update(pairwise(sequence))
            if not pair_counts:
                break
            (left, right), frequency = pair_counts.most_common(1)[0]
            if frequency < min_pair_frequency:
                break

            merged = left + right
            sequences = [self._merge_pair(sequence, left, right, merged) for sequence in sequences]
            self.merges.append(Merge(left, right, merged))
            vocabulary.add(merged)
            history.append(
                {
                    "step": len(self.merges),
                    "left": left,
                    "right": right,
                    "merged": merged,
                    "frequency": frequency,
                }
            )

        ordered_base = sorted(self.base_characters)
        ordered_merged = list(
            dict.fromkeys(
                merge.merged for merge in self.merges if merge.merged not in self.base_characters
            )
        )
        self.id_to_token = list(self.SPECIAL_TOKENS) + ordered_base + ordered_merged
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}
        return history

    @staticmethod
    def _merge_pair(sequence: list[str], left: str, right: str, merged: str) -> list[str]:
        result: list[str] = []
        index = 0
        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == left
                and sequence[index + 1] == right
            ):
                result.append(merged)
                index += 2
            else:
                result.append(sequence[index])
                index += 1
        return result

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        self._require_trained()
        pieces = [
            character if character in self.base_characters else self.UNK for character in text
        ]
        for merge in self.merges:
            pieces = self._merge_pair(pieces, merge.left, merge.right, merge.merged)
        ids = [self.token_to_id.get(piece, self.unk_id) for piece in pieces]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = True) -> str:
        self._require_trained()
        pieces: list[str] = []
        for token_id in token_ids:
            if not 0 <= int(token_id) < self.vocab_size:
                pieces.append(self.UNK)
                continue
            token = self.id_to_token[int(token_id)]
            if skip_special and token in self.SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces)

    def token_label(self, token_id: int) -> str:
        token = self.id_to_token[token_id]
        return token.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "␠")

    def to_dict(self) -> dict[str, Any]:
        self._require_trained()
        return {
            "id_to_token": self.id_to_token,
            "base_characters": sorted(self.base_characters),
            "merges": [
                {"left": merge.left, "right": merge.right, "merged": merge.merged}
                for merge in self.merges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BPETokenizer:
        tokenizer = cls()
        tokenizer.id_to_token = list(data["id_to_token"])
        tokenizer.token_to_id = {token: index for index, token in enumerate(tokenizer.id_to_token)}
        tokenizer.base_characters = set(data["base_characters"])
        tokenizer.merges = [Merge(**item) for item in data["merges"]]
        tokenizer._require_trained()
        return tokenizer

    def _require_trained(self) -> None:
        if not self.id_to_token:
            raise RuntimeError("Tokenizer has not been trained")
