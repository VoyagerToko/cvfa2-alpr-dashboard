from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class LabelEncoder:
    vocab: str
    blank_token: str = field(init=False, default="<BLANK>")
    idx_to_char: list[str] = field(init=False, default_factory=list)
    char_to_idx: dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.idx_to_char = [self.blank_token] + list(self.vocab)
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.vocab)}

    @property
    def blank_index(self) -> int:
        return 0

    @property
    def vocab_size(self) -> int:
        return len(self.idx_to_char)

    def encode(self, text: str) -> torch.Tensor:
        encoded = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.idx_to_char[idx] for idx in token_ids if idx > 0 and idx < self.vocab_size)

    def ctc_greedy_decode(self, logits: torch.Tensor) -> list[str]:
        """
        Decode CTC logits of shape (T, B, C) into a batch of strings.
        """
        token_ids = logits.argmax(dim=-1).transpose(0, 1).tolist()
        decoded: list[str] = []
        for seq in token_ids:
            prev = -1
            merged: list[int] = []
            for idx in seq:
                if idx != prev and idx != self.blank_index:
                    merged.append(idx)
                prev = idx
            decoded.append(self.decode(merged))
        return decoded
