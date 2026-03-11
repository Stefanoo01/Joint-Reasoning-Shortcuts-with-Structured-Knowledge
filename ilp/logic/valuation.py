from __future__ import annotations
from typing import List
from ilp.logic.atoms import Atom

def build_a0(
    n: int,
    atom_to_idx: dict[Atom, int],
    B: list[Atom],
    bot_idx: int,
) -> list[float]:
    """
    Build the initial valuation a0 in [0,1]^n from background facts B.

    Rules:
      - a0[i] = 1.0 if the corresponding ground atom is in B
      - a0[i] = 0.0 otherwise
      - a0[bot_idx] must always be 0.0
    """
    a0 = [0.0] * n

    for fact in B:
        if fact not in atom_to_idx:
            raise KeyError(f"Background fact not in language G: {fact}")
        a0[atom_to_idx[fact]] = 1.0

    a0[bot_idx] = 0.0
    return a0