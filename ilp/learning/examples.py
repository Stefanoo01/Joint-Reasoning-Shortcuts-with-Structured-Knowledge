from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

from ilp.logic.atoms import Atom
from ilp.learning.data import Targets, build_targets_from_positives


@dataclass(frozen=True)
class Example:
    """
    One training example / episode:
      - hard facts (certain background)
      - soft facts (probabilistic observations from concept extractor)
      - targets for the task predicate (pos/neg indices in G)
    """
    hard_facts: List[Atom]
    soft_facts: List[Tuple[Atom, float]]
    targets: Targets


def build_example_from_positives(
    *,
    atom_to_idx: Dict[Atom, int],
    constants: List[str],
    pred_name: str,
    arity: int,
    positive_atoms: List[Atom],
    hard_facts: List[Atom],
    soft_facts: List[Tuple[Atom, float]] | None = None,
) -> Example:
    """
    Build an Example with Targets computed from the positives for this example.
    Negatives are all other ground atoms of pred_name/arity (closed world).
    """
    if soft_facts is None:
        soft_facts = []

    targets = build_targets_from_positives(
        atom_to_idx=atom_to_idx,
        constants=constants,
        pred_name=pred_name,
        arity=arity,
        positive_atoms=positive_atoms,
    )
    return Example(hard_facts=hard_facts, soft_facts=soft_facts, targets=targets)