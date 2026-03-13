from __future__ import annotations

import torch

from ilp.logic.atoms import Atom, BOT, Predicate
from ilp.logic.language import LanguageSpec, build_ground_atoms, build_index
from ilp.logic.templates import RuleTemplate, ProgramTemplate

from ilp.learning.bias import BiasConfig
from ilp.learning.build_program import build_clause_sets_for_program, build_caches_with_bias
from ilp.learning.model import ProgramLearner
from ilp.learning.examples import build_example_from_positives
from ilp.learning.trainer import TrainConfig, train_program_examples
from ilp.logic.valuation_soft import build_a0_from_facts

def _setup():
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
    return C, atom_to_idx, B


def test_build_example_even():
    C, atom_to_idx, B = _setup()

    ex = build_example_from_positives(
        atom_to_idx=atom_to_idx,
        constants=C,
        pred_name="even",
        arity=1,
        positive_atoms=[Atom("even", ("0",)), Atom("even", ("2",)), Atom("even", ("4",))],
        hard_facts=B,
        soft_facts=[],
    )

    assert len(ex.hard_facts) == len(B)
    assert len(ex.targets.pos_idx) == 3
    assert len(ex.targets.neg_idx) == 3

def test_train_program_examples_smoke():
    # --- Setup toy language with succ2 as latent aux ---
    C = ["0", "1", "2", "3", "4", "5"]
    predicates = [
        Predicate("zero", 1, "E"),
        Predicate("succ", 2, "E"),
        Predicate("even", 1, "I"),
        Predicate("succ2", 2, "I"),
    ]

    target_pred = next(p for p in predicates if p.name == "even" and p.arity == 1)
    succ2_pred = next(p for p in predicates if p.name == "succ2" and p.arity == 2)

    spec = LanguageSpec(constants=C, predicates=predicates)
    G = build_ground_atoms(spec)
    atom_to_idx, idx_to_atom, bot_idx = build_index(G)
    n = len(G)

    B = [
        Atom("zero", ("0",)),
        Atom("succ", ("0", "1")),
        Atom("succ", ("1", "2")),
        Atom("succ", ("2", "3")),
        Atom("succ", ("3", "4")),
        Atom("succ", ("4", "5")),
    ]

    # Π templates
    tau1_even = RuleTemplate(v=0, int_flag=0)
    tau2_even = RuleTemplate(v=1, int_flag=1)
    tau_succ2 = RuleTemplate(v=1, int_flag=0)

    Pi = ProgramTemplate(
        aux_predicates=[succ2_pred],
        rules={
            ("even", 1): (tau1_even, tau2_even),
            ("succ2", 2): (tau_succ2, tau_succ2),
        },
        T=4,
    )

    # Clause sets + bias
    clause_sets = build_clause_sets_for_program(
        predicates=predicates,
        target_pred=target_pred,
        program=Pi,
    )

    bias = BiasConfig(
        allowed_body_preds={
            ("succ2", 2): {"succ"},
            ("even", 1): {"zero", "succ2", "even"},
        },
        require_recursive={},
        require_body_connected=True,
    )

    caches, clause_texts = build_caches_with_bias(
        clause_sets=clause_sets,
        constants=C,
        atom_to_idx=atom_to_idx,
        n=n,
        bot_idx=bot_idx,
        bias=bias,
        require_recursive_on_C2={("even", 1): True},
    )

    learner = ProgramLearner(caches)

    # Examples: 2 identical examples is fine for smoke
    positives = [Atom("even", ("0",)), Atom("even", ("2",)), Atom("even", ("4",))]

    ex1 = build_example_from_positives(
        atom_to_idx=atom_to_idx,
        constants=C,
        pred_name="even",
        arity=1,
        positive_atoms=positives,
        hard_facts=B,
        soft_facts=[],
    )
    ex2 = build_example_from_positives(
        atom_to_idx=atom_to_idx,
        constants=C,
        pred_name="even",
        arity=1,
        positive_atoms=positives,
        hard_facts=B,
        soft_facts=[],
    )

    cfg = TrainConfig(
        epochs=40,              # small for smoke
        lr=5e-2,
        temperature_start=2.0,
        temperature_end=0.5,
        entropy_coeff=1e-3,
        log_every=9999,         # avoid spam in tests
    )

    # If this runs without exceptions, the test passes
    train_program_examples(
        learner=learner,
        examples=[ex1, ex2],
        atom_to_idx=atom_to_idx,
        n=n,
        bot_idx=bot_idx,
        T=Pi.T,
        cfg=cfg,
        clause_texts=None,
        device=torch.device("cpu"),
    )


def test_program_learner_apply_updates_clause_caches():
    C = ["0", "1", "2", "3", "4", "5"]
    predicates = [
        Predicate("zero", 1, "E"),
        Predicate("succ", 2, "E"),
        Predicate("even", 1, "I"),
    ]

    target_pred = next(p for p in predicates if p.name == "even" and p.arity == 1)
    spec = LanguageSpec(constants=C, predicates=predicates)
    G = build_ground_atoms(spec)
    atom_to_idx, _, bot_idx = build_index(G)
    n = len(G)

    tau_even = RuleTemplate(v=0, int_flag=0)
    Pi = ProgramTemplate(
        aux_predicates=[],
        rules={("even", 1): (tau_even, tau_even)},
        T=1,
    )

    clause_sets = build_clause_sets_for_program(
        predicates=predicates,
        target_pred=target_pred,
        program=Pi,
    )

    bias = BiasConfig(
        allowed_body_preds={("even", 1): {"zero"}},
        require_recursive={},
        require_body_connected=True,
    )

    caches, _ = build_caches_with_bias(
        clause_sets=clause_sets,
        constants=C,
        atom_to_idx=atom_to_idx,
        n=n,
        bot_idx=bot_idx,
        bias=bias,
        require_recursive_on_C2={},
    )

    learner = ProgramLearner(caches)
    before = {
        key: (cache.X1, cache.X2)
        for key, cache in learner.caches.items()
    }

    learner._apply(lambda t: t.clone())

    for key, cache in learner.caches.items():
        old_x1, old_x2 = before[key]
        assert cache.X1 is not old_x1
        assert cache.X2 is not old_x2
        assert torch.equal(cache.X1, old_x1)
        assert torch.equal(cache.X2, old_x2)


def test_program_learner_batched_inference_matches_single():
    C = ["0", "1", "2", "3", "4", "5"]
    predicates = [
        Predicate("zero", 1, "E"),
        Predicate("succ", 2, "E"),
        Predicate("even", 1, "I"),
        Predicate("succ2", 2, "I"),
    ]

    target_pred = next(p for p in predicates if p.name == "even" and p.arity == 1)
    succ2_pred = next(p for p in predicates if p.name == "succ2" and p.arity == 2)

    spec = LanguageSpec(constants=C, predicates=predicates)
    G = build_ground_atoms(spec)
    atom_to_idx, _, bot_idx = build_index(G)
    n = len(G)

    tau1_even = RuleTemplate(v=0, int_flag=0)
    tau2_even = RuleTemplate(v=1, int_flag=1)
    tau_succ2 = RuleTemplate(v=1, int_flag=0)

    Pi = ProgramTemplate(
        aux_predicates=[succ2_pred],
        rules={
            ("even", 1): (tau1_even, tau2_even),
            ("succ2", 2): (tau_succ2, tau_succ2),
        },
        T=4,
    )

    clause_sets = build_clause_sets_for_program(
        predicates=predicates,
        target_pred=target_pred,
        program=Pi,
    )

    bias = BiasConfig(
        allowed_body_preds={
            ("succ2", 2): {"succ"},
            ("even", 1): {"zero", "succ2", "even"},
        },
        require_recursive={},
        require_body_connected=True,
    )

    caches, _ = build_caches_with_bias(
        clause_sets=clause_sets,
        constants=C,
        atom_to_idx=atom_to_idx,
        n=n,
        bot_idx=bot_idx,
        bias=bias,
        require_recursive_on_C2={("even", 1): True},
    )

    learner = ProgramLearner(caches)

    ex_a = build_a0_from_facts(
        n=n,
        atom_to_idx=atom_to_idx,
        bot_idx=bot_idx,
        hard_facts=[
            Atom("zero", ("0",)),
            Atom("succ", ("0", "1")),
            Atom("succ", ("1", "2")),
            Atom("succ", ("2", "3")),
        ],
        soft_facts=[],
    )
    ex_b = build_a0_from_facts(
        n=n,
        atom_to_idx=atom_to_idx,
        bot_idx=bot_idx,
        hard_facts=[
            Atom("zero", ("0",)),
            Atom("succ", ("0", "1")),
            Atom("succ", ("1", "2")),
            Atom("succ", ("2", "3")),
            Atom("succ", ("3", "4")),
        ],
        soft_facts=[],
    )

    batch = torch.stack([ex_a, ex_b], dim=0)
    single = torch.stack(
        [
            learner.infer_T_paper(ex_a, T=Pi.T, temperature=1.0, fast=True),
            learner.infer_T_paper(ex_b, T=Pi.T, temperature=1.0, fast=True),
        ],
        dim=0,
    )
    batched = learner.infer_T_paper(batch, T=Pi.T, temperature=1.0, fast=True)

    assert batched.shape == single.shape
    assert torch.allclose(batched, single, atol=1e-6)


if __name__ == "__main__":
    test_build_example_even()
    test_train_program_examples_smoke()
    test_program_learner_apply_updates_clause_caches()
    test_program_learner_batched_inference_matches_single()
    print("✅ tests/test_examples.py passed")
