from __future__ import annotations

import torch

from ilp.logic.atoms import Atom
from configs.toy_even import make_config
from ilp.learning.system_builder import build_system_from_config
from ilp.learning.examples import build_example_from_positives
from ilp.learning.trainer import TrainConfig, train_program_examples
from ilp.learning.trainer import extract_hard_program  # se lo hai già, altrimenti commenta


def main() -> None:
    # 1) Load task config
    cfg = make_config()

    # 2) Build system (language, G, mappings, Π, caches, learner)
    bundle = build_system_from_config(cfg)

    atom_to_idx = bundle.atom_to_idx
    bot_idx = bundle.bot_idx
    n = len(bundle.G)

    # 3) Build background facts B (toy-specific data, NOT config)
    B = [
        Atom("zero", ("0",)),
        Atom("succ", ("0", "1")),
        Atom("succ", ("1", "2")),
        Atom("succ", ("2", "3")),
        Atom("succ", ("3", "4")),
        Atom("succ", ("4", "5")),
    ]

    # 4) Build examples (targets are per-example)
    positives = [Atom("even", ("0",)), Atom("even", ("2",)), Atom("even", ("4",))]

    ex1 = build_example_from_positives(
        atom_to_idx=atom_to_idx,
        constants=cfg.constants,
        pred_name="even",
        arity=1,
        positive_atoms=positives,
        hard_facts=B,
        soft_facts=[],
    )
    # For now, just duplicate to test batching
    ex2 = build_example_from_positives(
        atom_to_idx=atom_to_idx,
        constants=cfg.constants,
        pred_name="even",
        arity=1,
        positive_atoms=positives,
        hard_facts=B,
        soft_facts=[],
    )

    examples = [ex1, ex2]

    # 5) Train
    train_cfg = TrainConfig(
        epochs=200,
        lr=5e-2,
        temperature_start=2.0,
        temperature_end=0.2,
        entropy_coeff=1e-3,
        log_every=25,
    )

    train_program_examples(
        learner=bundle.learner,
        examples=examples,
        atom_to_idx=atom_to_idx,
        n=n,
        bot_idx=bot_idx,
        T=bundle.program.T,
        cfg=train_cfg,
        clause_texts=bundle.clause_texts,
        device=torch.device("cpu"),
    )

    # 6) Print hard program (optional)
    try:
        hard = extract_hard_program(bundle.learner, temperature=0.2)
        print("\n=== HARD PROGRAM ===")
        for key, (j, k, prob) in hard.items():
            print(f"{key} -> (j={j}, k={k}) prob={prob:.3f}")
            print("  C1:", bundle.clause_texts[key][0][j])
            print("  C2:", bundle.clause_texts[key][1][k])
    except Exception:
        pass


if __name__ == "__main__":
    main()