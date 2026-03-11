from __future__ import annotations

import argparse
import torch

from ilp.logic.atoms import Atom
from configs.toy_sum_parity import make_config
from ilp.learning.system_builder import build_system_from_config
from ilp.learning.examples import build_example_from_positives
from ilp.learning.trainer import TrainConfig, train_program_examples
from ilp.learning.trainer import extract_hard_program

from data.adapters import ToySumParityAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toy sum-parity experiment")
    parser.add_argument(
        "--mode", type=str, default="relaxed", choices=["relaxed", "guided"],
        help="Bias mode: 'relaxed' (produces shortcut) or 'guided' (finds correct program)",
    )
    parser.add_argument("--epochs", type=int, default=800)
    args = parser.parse_args()

    # 1) Load task config
    cfg = make_config(mode=args.mode)
    print(f"Mode: {args.mode}")

    # 2) Build system
    bundle = build_system_from_config(cfg)

    atom_to_idx = bundle.atom_to_idx
    bot_idx = bundle.bot_idx
    n = len(bundle.G)

    # 3) Build examples via adapter
    adapter = ToySumParityAdapter()
    examples = adapter.build_examples(bundle)
    print(f"Built {len(examples)} examples")

    # 4) Train
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=5e-2,
        temperature_start=2.0,
        temperature_end=0.2,
        entropy_coeff=1e-3,
        log_every=100,
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

    # 5) Print hard program
    try:
        hard = extract_hard_program(bundle.learner, temperature=0.2)
        print("\n=== HARD PROGRAM ===")
        for key, (j, k, prob) in hard.items():
            print(f"{key} -> (j={j}, k={k}) prob={prob:.3f}")
            print("  C1:", bundle.clause_texts[key][0][j])
            print("  C2:", bundle.clause_texts[key][1][k])
    except Exception as e:
        print(f"Could not extract hard program: {e}")


if __name__ == "__main__":
    main()
