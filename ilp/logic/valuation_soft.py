from __future__ import annotations

from typing import Dict, Iterable, Tuple
import torch

from ilp.logic.atoms import Atom


def build_a0_from_facts(
    *,
    n: int,
    atom_to_idx: Dict[Atom, int],
    bot_idx: int,
    hard_facts: Iterable[Atom] = (),
    soft_facts: Iterable[Tuple[Atom, float]] = (),
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build initial valuation a0 in [0,1]^n from:
      - hard_facts: atoms forced to 1
      - soft_facts: (atom, prob) assigned in [0,1]
    Rules:
      - BOT is always 0
      - unknown atoms are ignored
      - if an atom appears multiple times in soft_facts, we take max(prob)
      - hard_facts override everything (set to 1)
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be a positive int, got {n}")
    if not isinstance(bot_idx, int) or bot_idx < 0 or bot_idx >= n:
        raise ValueError(f"bot_idx must be in [0, {n-1}], got {bot_idx}")

    if device is None:
        device = torch.device("cpu")

    a0 = torch.zeros(n, dtype=torch.float32, device=device)

    # 1) Apply soft facts first (max-merge)
    for at, p in soft_facts:
        idx = atom_to_idx.get(at, None)
        if idx is None or idx == bot_idx:
            continue

        pp = float(p)
        if pp < 0.0:
            pp = 0.0
        elif pp > 1.0:
            pp = 1.0

        # max merge
        if pp > float(a0[idx].item()):
            a0[idx] = pp

    # 2) Apply hard facts (override)
    for at in hard_facts:
        idx = atom_to_idx.get(at, None)
        if idx is None or idx == bot_idx:
            continue
        a0[idx] = 1.0

    # 3) Ensure BOT is always 0
    a0[bot_idx] = 0.0
    return a0

from typing import Optional
import torch


def build_a0_from_indexed_facts(
    *,
    n: int,
    bot_idx: int,
    soft_idx: torch.Tensor,  # [m] long
    soft_val: torch.Tensor,  # [m] float, differentiable
    hard_idx: Optional[torch.Tensor] = None,  # [h] long
) -> torch.Tensor:
    """
    Differentiable a0 builder.
    - soft facts provided as indices + values (keeps gradient)
    - hard facts override to 1.0
    - BOT forced to 0.0
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if bot_idx < 0 or bot_idx >= n:
        raise ValueError("bot_idx out of range")
    if soft_idx.dtype != torch.long:
        raise ValueError("soft_idx must be torch.long")
    if soft_val.dtype not in (torch.float32, torch.float64):
        raise ValueError("soft_val must be float tensor")
    if soft_idx.dim() != 1 or soft_val.dim() != 1:
        raise ValueError("soft_idx and soft_val must be 1D")
    if soft_idx.numel() != soft_val.numel():
        raise ValueError("soft_idx and soft_val must have same length")

    device = soft_val.device
    a0_soft = torch.zeros(n, dtype=soft_val.dtype, device=device)

    # scatter max for soft facts
    # (PyTorch supports scatter_reduce_ in recent versions)
    a0_soft.scatter_reduce_(0, soft_idx, soft_val, reduce="amax", include_self=True)
    a0 = a0_soft.clone()

    # hard override
    if hard_idx is not None:
        if hard_idx.dtype != torch.long or hard_idx.dim() != 1:
            raise ValueError("hard_idx must be 1D torch.long")
        a0[hard_idx] = 1.0

    # BOT always 0
    a0[bot_idx] = 0.0
    return a0