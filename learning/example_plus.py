from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from logic.atoms import Atom
from learning.data import Targets


@dataclass(frozen=True)
class ExamplePlus:
    """
    One training example with:
      - hard facts (certain background)
      - soft facts (probabilistic observations from concept extractor)
      - task targets (e.g., sum_is/1 for MNIST-Addition)
      - optional concept targets (supervised CBM phase), keyed by predicate name
        e.g., {"digit": Targets(...)}.
    """
    hard_facts: List[Atom]
    soft_facts: List[Tuple[Atom, float]]
    task_targets: Targets
    concept_targets: Optional[Dict[str, Targets]] = None