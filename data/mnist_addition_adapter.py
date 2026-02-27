from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random

from logic.atoms import Atom
from learning.data import Targets
from learning.example_plus import ExamplePlus


def _build_sum_targets(
    atom_to_idx: Dict[Atom, int],
    sums: List[str],
    true_sum: int,
) -> Targets:
    all_atoms = [Atom("sum_is", (s,)) for s in sums]
    all_idx = [atom_to_idx[a] for a in all_atoms]

    pos_atom = Atom("sum_is", (str(true_sum),))
    pos_idx = [atom_to_idx[pos_atom]]
    neg_idx = [i for i in all_idx if i != pos_idx[0]]
    return Targets(all_idx=all_idx, pos_idx=pos_idx, neg_idx=neg_idx)


def _build_digit_pos_targets(
    atom_to_idx: Dict[Atom, int],
    pos: str,
    digits: List[str],
    true_digit: int,
) -> Targets:
    all_atoms = [Atom("digit", (pos, d)) for d in digits]
    all_idx = [atom_to_idx[a] for a in all_atoms]

    pos_atom = Atom("digit", (pos, str(true_digit)))
    pos_idx = [atom_to_idx[pos_atom]]
    neg_idx = [i for i in all_idx if i != pos_idx[0]]
    return Targets(all_idx=all_idx, pos_idx=pos_idx, neg_idx=neg_idx)


def _one_hot_probs(true_digit: int) -> List[float]:
    p = [0.0] * 10
    p[true_digit] = 1.0
    return p


def _noisy_probs(true_digit: int, noise: float) -> List[float]:
    """
    Simple noise model:
      - put (1-noise) mass on the true digit
      - spread noise uniformly across the others
    """
    if noise <= 0.0:
        return _one_hot_probs(true_digit)
    if noise >= 1.0:
        # uniform
        return [0.1] * 10

    other = noise / 9.0
    p = [other] * 10
    p[true_digit] = 1.0 - noise
    return p


@dataclass(frozen=True)
class MNISTAdditionStubAdapter:
    """
    Produces synthetic MNIST-Addition examples in the same format we will use with real data.
    This adapter is for plumbing/testing only (no torchvision/RSBench dependency).

    If supervised_concepts=True, it fills concept_targets for digits.
    """
    num_examples: int = 32
    noise: float = 0.0
    seed: int = 0
    supervised_concepts: bool = True

    def build_examples(self, *, atom_to_idx: Dict[Atom, int]) -> List[ExamplePlus]:
        rng = random.Random(self.seed)
        
        digits = [str(d) for d in range(10)]
        sums = [str(s) for s in range(19)]

        out: List[ExamplePlus] = []
        for _ in range(self.num_examples):
            d1 = rng.randrange(10)
            d2 = rng.randrange(10)
            s = d1 + d2

            # soft facts from "concept extractor"
            p1 = _noisy_probs(d1, self.noise)
            p2 = _noisy_probs(d2, self.noise)

            soft_facts: List[Tuple[Atom, float]] = []
            for d, prob in enumerate(p1):
                soft_facts.append((Atom("digit", ("d1", str(d))), float(prob)))
            for d, prob in enumerate(p2):
                soft_facts.append((Atom("digit", ("d2", str(d))), float(prob)))

            # task targets for sum_is/1
            task_targets = _build_sum_targets(atom_to_idx, sums, true_sum=s)

            # concept targets (optional, supervised CBM phase)
            concept_targets = None
            if self.supervised_concepts:
                concept_targets = {
                    "digit_d1": _build_digit_pos_targets(atom_to_idx, "d1", digits, true_digit=d1),
                    "digit_d2": _build_digit_pos_targets(atom_to_idx, "d2", digits, true_digit=d2),
                }

            ex = ExamplePlus(
                hard_facts=[],                 # no hard background for now
                soft_facts=soft_facts,
                task_targets=task_targets,
                concept_targets=concept_targets,
            )
            out.append(ex)

        return out