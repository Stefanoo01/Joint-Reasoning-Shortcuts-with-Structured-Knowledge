from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Optional

from logic.atoms import Atom
from learning.examples import Example, build_example_from_positives
from learning.system_builder import SystemBundle


class ExampleAdapter(Protocol):
    def build_examples(self, bundle: SystemBundle) -> List[Example]:
        ...


@dataclass(frozen=True)
class ToyEvenAdapter:
    """
    Produces a small fixed dataset of Examples for the toy_even task.
    """
    num_examples: int = 2

    def build_examples(self, bundle: SystemBundle) -> List[Example]:
        C = bundle.spec.constants
        atom_to_idx = bundle.atom_to_idx

        B = [
            Atom("zero", ("0",)),
            Atom("succ", ("0", "1")),
            Atom("succ", ("1", "2")),
            Atom("succ", ("2", "3")),
            Atom("succ", ("3", "4")),
            Atom("succ", ("4", "5")),
        ]

        positives = [Atom("even", ("0",)), Atom("even", ("2",)), Atom("even", ("4",))]

        out: List[Example] = []
        for _ in range(self.num_examples):
            out.append(
                build_example_from_positives(
                    atom_to_idx=atom_to_idx,
                    constants=C,
                    pred_name="even",
                    arity=1,
                    positive_atoms=positives,
                    hard_facts=B,
                    soft_facts=[],
                )
            )
        return out