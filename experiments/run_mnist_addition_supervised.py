from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from logic.atoms import Atom
from logic.valuation_soft import build_a0_from_indexed_facts
from configs.mnist_addition import make_config
from learning.system_builder import build_system_from_config
from learning.model import bce_pos_neg
from models.cbm_mnist import MNISTDigitCNN

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


@dataclass
class PairSample:
    img1: torch.Tensor
    img2: torch.Tensor
    d1: int
    d2: int
    s: int


def build_add_truth_table_hard_idx(atom_to_idx) -> torch.Tensor:
    """
    hard facts for add(A,B,S) for all A,B in 0..9.
    """
    hard_atoms = []
    for a in range(10):
        for b in range(10):
            s = a + b
            hard_atoms.append(Atom("add", (str(a), str(b), str(s))))
    hard_idx = torch.tensor([atom_to_idx[a] for a in hard_atoms], dtype=torch.long)
    return hard_idx


def build_digit_soft_idx(atom_to_idx) -> torch.Tensor:
    """
    soft facts indices for digit(d1,0..9) + digit(d2,0..9) in fixed order.
    """
    atoms = []
    for d in range(10):
        atoms.append(Atom("digit", ("d1", str(d))))
    for d in range(10):
        atoms.append(Atom("digit", ("d2", str(d))))
    return torch.tensor([atom_to_idx[a] for a in atoms], dtype=torch.long)


def build_sum_targets(atom_to_idx, cfg_constants: List[str], true_sum: int):
    # targets for sum_is/1: positive is sum_is(true_sum), negatives are other sum_is
    sums = [str(s) for s in range(19)]
    from learning.data import build_targets_from_positives
    return build_targets_from_positives(
        atom_to_idx=atom_to_idx,
        constants=cfg_constants,
        pred_name="sum_is",
        arity=1,
        positive_atoms=[Atom("sum_is", (str(true_sum),))],
    )


def main() -> None:
    device = torch.device("cuda")

    # 1) Build system (ILP)
    cfg = make_config()
    bundle = build_system_from_config(cfg)
    learner = bundle.learner.to(device)

    n = len(bundle.G)
    bot_idx = bundle.bot_idx
    atom_to_idx = bundle.atom_to_idx

    # 2) Precompute indices for facts
    hard_idx_add = build_add_truth_table_hard_idx(atom_to_idx).to(device)
    soft_idx_digit = build_digit_soft_idx(atom_to_idx).to(device)

    # 3) CBM model
    cbm = MNISTDigitCNN().to(device)

    # 4) Optimizer (CBM + ILP)
    opt = torch.optim.Adam(list(cbm.parameters()) + list(learner.parameters()), lr=1e-3)

    tfm = transforms.Compose([transforms.ToTensor()])  # [1,28,28] in [0,1]
    train_ds = MNIST(root="data_root", train=True, download=True, transform=tfm)

    # Simple pair dataloader: sample two random indices per batch element
    batch_size = 16
    rng = random.Random(0)

    def sample_batch() -> List[PairSample]:
        out = []
        for _ in range(batch_size):
            i1 = rng.randrange(len(train_ds))
            i2 = rng.randrange(len(train_ds))
            img1, d1 = train_ds[i1]
            img2, d2 = train_ds[i2]
            out.append(PairSample(img1.to(device), img2.to(device), int(d1), int(d2), int(d1) + int(d2)))
        return out

    # 6) Training loop (supervised CBM)
    epochs = 3
    steps_per_epoch = 100
    lambda_concept = 1.0  # concept supervision weight

    for ep in range(epochs):
        for step in range(steps_per_epoch):
            batch = sample_batch()

            imgs1 = torch.stack([b.img1 for b in batch], dim=0)  # [B,1,28,28]
            imgs2 = torch.stack([b.img2 for b in batch], dim=0)
            d1_true = torch.tensor([b.d1 for b in batch], dtype=torch.long, device=device)
            d2_true = torch.tensor([b.d2 for b in batch], dtype=torch.long, device=device)

            # CBM forward
            logits1 = cbm(imgs1)  # [B,10]
            logits2 = cbm(imgs2)  # [B,10]
            probs1 = torch.softmax(logits1, dim=1)  # [B,10]
            probs2 = torch.softmax(logits2, dim=1)

            # Concept loss (supervised)
            loss_concepts = (F.cross_entropy(logits1, d1_true) + F.cross_entropy(logits2, d2_true)) / 2.0

            # Task loss via ILP (loop over batch, simple & safe)
            loss_task = 0.0
            for i, b in enumerate(batch):
                soft_val = torch.cat([probs1[i], probs2[i]], dim=0)  # [20]
                a0 = build_a0_from_indexed_facts(
                    n=n,
                    bot_idx=bot_idx,
                    soft_idx=soft_idx_digit,
                    soft_val=soft_val,
                    hard_idx=hard_idx_add,
                )
                aT = learner.infer_T_paper(a0, T=bundle.program.T, temperature=1.0, fast=True)

                targets = build_sum_targets(atom_to_idx, cfg.constants, true_sum=b.s)
                loss_task = loss_task + bce_pos_neg(aT, targets.pos_idx, targets.neg_idx)

            loss_task = loss_task / batch_size

            loss = loss_task + lambda_concept * loss_concepts

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 20 == 0:
                print(f"[ep {ep} step {step}] loss_task={float(loss_task.item()):.4f} "
                      f"loss_concepts={float(loss_concepts.item()):.4f} total={float(loss.item()):.4f}")

    print("Done.")


if __name__ == "__main__":
    main()