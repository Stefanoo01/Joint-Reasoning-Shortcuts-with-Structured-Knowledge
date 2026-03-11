from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Optional

from ilp.logic.atoms import Atom
from ilp.learning.examples import Example, build_example_from_positives
from ilp.learning.system_builder import SystemBundle


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


@dataclass(frozen=True)
class ToySumParityAdapter:
    """
    Produces examples for the toy_sum_parity task.
    One example per digit pair (a, b) where a, b in {0..4}.
    """

    def build_examples(self, bundle: SystemBundle) -> List[Example]:
        C = bundle.spec.constants
        atom_to_idx = bundle.atom_to_idx

        # Background parity facts (same for all examples)
        parity_facts = [
            Atom("is_even", (str(d),)) for d in range(6) if d % 2 == 0
        ] + [
            Atom("is_odd", (str(d),)) for d in range(6) if d % 2 == 1
        ]

        out: List[Example] = []
        for a in range(6):
            for b in range(6):
                sa, sb = str(a), str(b)

                # Per-example facts: which digits are given
                hard_facts = parity_facts + [
                    Atom("digit1", (sa,)),
                    Atom("digit2", (sb,)),
                ]

                # Target: sum_even(digit1_value) is positive iff sum is even
                s = a + b
                if s % 2 == 0:
                    positives = [Atom("sum_even", (sa,))]
                else:
                    positives = []  # no positives when sum is odd

                out.append(
                    build_example_from_positives(
                        atom_to_idx=atom_to_idx,
                        constants=C,
                        pred_name="sum_even",
                        arity=1,
                        positive_atoms=positives,
                        hard_facts=hard_facts,
                        soft_facts=[],
                    )
                )
        return out