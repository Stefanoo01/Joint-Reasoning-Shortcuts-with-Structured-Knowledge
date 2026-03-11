from __future__ import annotations

import argparse
import torch

from configs.registry import TASK_CONFIGS
from ilp.learning.system_builder import build_system_from_config
from ilp.learning.trainer import TrainConfig, train_program_examples

from data.adapters import ToyEvenAdapter, ToySumParityAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=TASK_CONFIGS.keys())
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # 1) Load config
    cfg = TASK_CONFIGS[args.task]()

    # 2) Build system from config
    bundle = build_system_from_config(cfg)

    # 3) Build examples via adapter
    if args.task == "toy_even":
        adapter = ToyEvenAdapter(num_examples=2)
    elif args.task == "toy_sum_parity":
        adapter = ToySumParityAdapter()
    else:
        raise RuntimeError("No adapter registered for this task yet")

    examples = adapter.build_examples(bundle)

    # 4) Train
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        temperature_start=2.0,
        temperature_end=0.2,
        entropy_coeff=1e-3,
        log_every=max(1, args.epochs // 8),
    )

    device = torch.device(args.device)

    train_program_examples(
        learner=bundle.learner,
        examples=examples,
        atom_to_idx=bundle.atom_to_idx,
        n=len(bundle.G),
        bot_idx=bundle.bot_idx,
        T=bundle.program.T,
        cfg=train_cfg,
        clause_texts=bundle.clause_texts,
        device=device,
    )


if __name__ == "__main__":
    main()