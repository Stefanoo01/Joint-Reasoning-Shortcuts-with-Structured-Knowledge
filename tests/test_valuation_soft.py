from __future__ import annotations

import torch

from ilp.logic.atoms import Atom, BOT, Predicate
from ilp.logic.language import LanguageSpec, build_ground_atoms, build_index
from ilp.logic.valuation_soft import build_a0_from_facts


def _setup_toy_even():
    C = ["0", "1", "2", "3", "4", "5"]
    predicates = [
        Predicate("zero", 1, "E"),
        Predicate("succ", 2, "E"),
        Predicate("even", 1, "I"),
    ]
    spec = LanguageSpec(constants=C, predicates=predicates)
    G = build_ground_atoms(spec)
    atom_to_idx, idx_to_atom, bot_idx = build_index(G)

    B = [
        Atom("zero", ("0",)),
        Atom("succ", ("0", "1")),
        Atom("succ", ("1", "2")),
        Atom("succ", ("2", "3")),
        Atom("succ", ("3", "4")),
        Atom("succ", ("4", "5")),
    ]
    return C, G, atom_to_idx, bot_idx, B


def test_hard_facts_only():
    C, G, atom_to_idx, bot_idx, B = _setup_toy_even()
    n = len(G)

    a0 = build_a0_from_facts(n=n, atom_to_idx=atom_to_idx, bot_idx=bot_idx, hard_facts=B, soft_facts=[])
    assert a0.shape == (n,)
    assert a0.dtype == torch.float32
    assert float(a0[bot_idx].item()) == 0.0

    assert float(a0[atom_to_idx[Atom("zero", ("0",))]].item()) == 1.0
    assert float(a0[atom_to_idx[Atom("succ", ("2", "3"))]].item()) == 1.0
    assert float(a0[atom_to_idx[Atom("zero", ("1",))]].item()) == 0.0
    assert float(a0[atom_to_idx[Atom("even", ("0",))]].item()) == 0.0


def test_soft_fact_assignment():
    C, G, atom_to_idx, bot_idx, B = _setup_toy_even()
    n = len(G)

    idx = atom_to_idx[Atom("even", ("2",))]
    a0 = build_a0_from_facts(
        n=n,
        atom_to_idx=atom_to_idx,
        bot_idx=bot_idx,
        hard_facts=B,
        soft_facts=[(Atom("even", ("2",)), 0.7)],
    )
    assert abs(float(a0[idx].item()) - 0.7) < 1e-6


def test_hard_overrides_soft():
    C, G, atom_to_idx, bot_idx, B = _setup_toy_even()
    n = len(G)

    # Put a hard fact also in soft with low prob -> should become 1.0
    at = Atom("zero", ("0",))
    idx = atom_to_idx[at]
    a0 = build_a0_from_facts(
        n=n,
        atom_to_idx=atom_to_idx,
        bot_idx=bot_idx,
        hard_facts=B,
        soft_facts=[(at, 0.2)],
    )
    assert float(a0[idx].item()) == 1.0


def test_soft_duplicates_take_max():
    C, G, atom_to_idx, bot_idx, B = _setup_toy_even()
    n = len(G)

    at = Atom("even", ("4",))
    idx = atom_to_idx[at]
    a0 = build_a0_from_facts(
        n=n,
        atom_to_idx=atom_to_idx,
        bot_idx=bot_idx,
        hard_facts=B,
        soft_facts=[(at, 0.2), (at, 0.9), (at, 0.4)],
    )
    assert abs(float(a0[idx].item()) - 0.9) < 1e-6


def test_bot_is_always_zero():
    C, G, atom_to_idx, bot_idx, B = _setup_toy_even()
    n = len(G)

    # Try to set BOT as hard and soft (should still be 0)
    a0 = build_a0_from_facts(
        n=n,
        atom_to_idx=atom_to_idx,
        bot_idx=bot_idx,
        hard_facts=list(B) + [BOT],
        soft_facts=[(BOT, 1.0)],
    )
    assert float(a0[bot_idx].item()) == 0.0


if __name__ == "__main__":
    # simple runner without pytest
    test_hard_facts_only()
    test_soft_fact_assignment()
    test_hard_overrides_soft()
    test_soft_duplicates_take_max()
    test_bot_is_always_zero()
    print("✅ tests/test_valuation_soft.py passed")