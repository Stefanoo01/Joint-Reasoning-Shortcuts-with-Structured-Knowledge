from __future__ import annotations

import sys
import os
import random
import statistics
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Mount RSBench in sys.path
RSBENCH_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "third_party", "rsbench-code", "rsseval", "rss"
))
if RSBENCH_DIR not in sys.path:
    sys.path.insert(0, RSBENCH_DIR)

# RSBench imports
from models.mnistcbm import get_parser
from datasets.halfmnist import HALFMNIST
from models.mnistcbm import MnistCBM

# Differentiable ILP imports
from ilp.logic.atoms import Atom
from ilp.logic.valuation_soft import build_a0_from_indexed_facts
from configs.half_mnist_addition import make_config
from ilp.learning.system_builder import build_system_from_config
from ilp.learning.model import bce_pos_neg
from ilp.learning.data import build_targets_from_positives_domains
from ilp.learning.trainer import linear_anneal


def build_add_truth_table_hard_idx(atom_to_idx) -> torch.Tensor:
    hard_atoms = []
    # HALFMNIST digits are 0..4
    for a in range(5):
        for b in range(5):
            s = a + b
            hard_atoms.append(Atom("add", (str(a), str(b), str(s))))
    hard_idx = torch.tensor([atom_to_idx[a] for a in hard_atoms], dtype=torch.long)
    return hard_idx

def build_digit12_soft_idx(atom_to_idx) -> torch.Tensor:
    atoms = []
    for d in range(5):
        atoms.append(Atom("digit1", (str(d),)))
    for d in range(5):
        atoms.append(Atom("digit2", (str(d),)))
    return torch.tensor([atom_to_idx[a] for a in atoms], dtype=torch.long)

def build_sum_targets(atom_to_idx, cfg, true_sum: int):
    domains = cfg.arg_domains[("sum_is", 1)]  # [[0,1,2,3,4,5,6,7,8]]
    return build_targets_from_positives_domains(
        atom_to_idx=atom_to_idx,
        pred_name="sum_is",
        domains=domains,
        positive_atoms=[Atom("sum_is", (str(true_sum),))],
    )

def build_sum_is_idx(atom_to_idx) -> torch.Tensor:
    # 0..8 to map directly index -> sum
    atoms = [Atom("sum_is", (str(s),)) for s in range(9)]
    return torch.tensor([atom_to_idx[a] for a in atoms], dtype=torch.long)

@torch.no_grad()
def evaluate(
    *,
    cbm: nn.Module,
    learner: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_atoms: int,
    bot_idx: int,
    T: int,
    soft_idx_digit: torch.Tensor,
    hard_idx_add: torch.Tensor,
    sum_is_idx: torch.Tensor,
    sum_targets_cache: list,
) -> dict:
    cbm.eval()
    learner.eval()

    total = 0
    correct_d1 = 0
    correct_d2 = 0
    correct_sum_cbm = 0
    correct_sum_ilp = 0
    loss_task_sum = 0.0
    loss_concepts_sum = 0.0
    
    # Track the shortcut mappings
    from collections import Counter
    shortcut_map = Counter()

    for imgs, targets, concepts in loader:
        B = imgs.size(0)
        imgs = imgs.to(device)
        d1_true_t = concepts[:, 0].long().to(device)
        d2_true_t = concepts[:, 1].long().to(device)
        s_true_list = targets.long().tolist()

        cbm_out = cbm(imgs)
        # raw logits for CE loss over concepts
        # cbm_out["CS"] has shape [B, 2, 5]
        logits1 = cbm_out["CS"][:, 0, :]
        logits2 = cbm_out["CS"][:, 1, :]
        
        # normalize pCS has shape [B, 2, 5], containing probabilities
        probs1 = cbm_out["pCS"][:, 0, :]
        probs2 = cbm_out["pCS"][:, 1, :]

        # concept accuracy
        d1_hat = torch.argmax(logits1, dim=1)
        d2_hat = torch.argmax(logits2, dim=1)
        correct_d1 += int(((d1_hat == d1_true_t) & (d1_true_t != -1)).sum().item())
        correct_d2 += int(((d2_hat == d2_true_t) & (d2_true_t != -1)).sum().item())
        valid_c_mask = (d1_true_t != -1)
        total_valid_c = int(valid_c_mask.sum().item())

        # cbm baseline sum
        sum_hat_cbm = (d1_hat + d2_hat).detach().cpu().tolist()

        # Track the mappings WRT the true sum
        for i in range(B):
            if s_true_list[i] in [0, 1, 5, 6]:
                shortcut_map[(s_true_list[i], int(d1_hat[i]), int(d2_hat[i]))] += 1

        loss_concepts = (F.cross_entropy(logits1, d1_true_t, ignore_index=-1) + F.cross_entropy(logits2, d2_true_t, ignore_index=-1)) / 2.0
        if torch.isnan(loss_concepts):
            loss_concepts = torch.tensor(0.0, device=device)
        loss_concepts_sum += float(loss_concepts.item()) * B

        for i in range(B):
            # pCS provides 5 values for digit1 and 5 for digit2 => 10 soft facts
            soft_val = torch.cat([probs1[i].squeeze(), probs2[i].squeeze()], dim=0)

            a0 = build_a0_from_indexed_facts(
                n=n_atoms,
                bot_idx=bot_idx,
                soft_idx=soft_idx_digit,
                soft_val=soft_val,
                hard_idx=hard_idx_add,
            )
            aT = learner.infer_T_paper(a0, T=T, temperature=1.0, fast=True)

            s = s_true_list[i]
            if sum_hat_cbm[i] == s:
                correct_sum_cbm += 1

            scores = aT[sum_is_idx]
            s_hat_ilp = int(torch.argmax(scores).item())
            if s_hat_ilp == s:
                correct_sum_ilp += 1

            ilp_targets = sum_targets_cache[s]
            loss_task_sum += float(bce_pos_neg(aT, ilp_targets.pos_idx, ilp_targets.neg_idx).item())

        total += B


    # Log the mappings sorted by True Sum
    print("\n--- Model Reasoning Shortcut Mappings ---")
    for (s_true, d1, d2), count in sorted(shortcut_map.items()):
        if count > 10: # Only print significant mappings
            print(f"When True Sum was {s_true}, CBM guessed digits: ({d1}, {d2}) [{count} times]")
    print("------------------------------------------")

    return {
        "loss_task": loss_task_sum / total,
        "loss_concepts": loss_concepts_sum / total,
        "acc_concepts": ((correct_d1 + correct_d2) / (2 * total)) if total > 0 else 0.0,
        "acc_sum_cbm": correct_sum_cbm / total,
        "acc_sum_ilp": correct_sum_ilp / total,
    }


def main():
    parser = get_parser()
    
    # Custom overrides for this specific script
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lambda_mode", type=str, choices=["fixed", "schedule"], default="schedule")
    parser.add_argument("--lam0", type=float, default=1.0)
    parser.add_argument("--lam1", type=float, default=0.2)
    parser.add_argument("--lam2", type=float, default=0.0)

    # We will pretend the user passed reasonable defaults for RSBench internally
    args, unknown = parser.parse_known_args()
    
    # Force RSBench args to valid choices for HALFMNIST
    args.dataset = "halfmnist"
    args.task = "addition"
    if not hasattr(args, "c_sup") or args.c_sup == 0:
        args.c_sup = 1.0  # 100% supervision on concepts

    random.seed(args.seed or 123)
    torch.manual_seed(args.seed or 123)
    device = torch.device("cuda")

    # 1. Datasets
    print("Loading RSBench HALFMNIST...")
    dataset = HALFMNIST(args)
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    dataset.print_stats()

    # 2. Extract RSBench configured backbone for CBM
    encoder, _ = dataset.get_backbone()
    
    # n_facts=5 means 5 classes (digits 0..4) per digit
    cbm = MnistCBM(encoder=encoder, n_images=2, args=args, n_facts=5, nr_classes=9).to(device)

    # 3. Build ILP
    cfg = make_config()
    bundle = build_system_from_config(cfg)
    learner = bundle.learner.to(device)

    n_atoms = len(bundle.G)
    bot_idx = bundle.bot_idx
    atom_to_idx = bundle.atom_to_idx

    hard_idx_add = build_add_truth_table_hard_idx(atom_to_idx).to(device)
    soft_idx_digit = build_digit12_soft_idx(atom_to_idx).to(device)  # digit1(0..4)+digit2(0..4) => length 10
    sum_is_idx = build_sum_is_idx(atom_to_idx).to(device)

    # Cache targets up to sum=8
    sum_targets_cache = [build_sum_targets(atom_to_idx, cfg, true_sum=s) for s in range(9)]

    opt = torch.optim.Adam(list(cbm.parameters()) + list(learner.parameters()), lr=1e-3)

    def lambda_for_epoch(ep: int) -> float:
        if args.lambda_mode == "fixed":
            return args.lam0
        if ep >= 0: return args.lam0
        if ep == 1: return args.lam1
        return args.lam2

    print("\nTraining...")
    for ep in range(args.epochs):
        lam = lambda_for_epoch(ep)
        print(f"Epoch {ep+1}/{args.epochs} | Lambda: {lam}")
        cbm.train()
        learner.train()

        total_loss = 0.0
        # iterate mini batches
        for imgs, targets, concepts in train_loader:
            B = imgs.size(0)
            imgs = imgs.to(device)
            d1_true_t = concepts[:, 0].long().to(device)
            d2_true_t = concepts[:, 1].long().to(device)
            s_true_list = targets.long().tolist()

            cbm_out = cbm(imgs)
            logits1 = cbm_out["CS"][:, 0, :]
            logits2 = cbm_out["CS"][:, 1, :]
            probs1 = cbm_out["pCS"][:, 0, :]
            probs2 = cbm_out["pCS"][:, 1, :]

            loss_concepts = (F.cross_entropy(logits1, d1_true_t, ignore_index=-1) + F.cross_entropy(logits2, d2_true_t, ignore_index=-1)) / 2.0
            if torch.isnan(loss_concepts):
                loss_concepts = torch.tensor(0.0, device=device)

            loss_task = 0.0
            for i in range(B):
                soft_val = torch.cat([probs1[i].squeeze(), probs2[i].squeeze()], dim=0)

                a0 = build_a0_from_indexed_facts(
                    n=n_atoms,
                    bot_idx=bot_idx,
                    soft_idx=soft_idx_digit,
                    soft_val=soft_val,
                    hard_idx=hard_idx_add,
                )

                aT = learner.infer_T_paper(a0, T=bundle.program.T, temperature=1.0, fast=True)

                s = s_true_list[i]
                ilp_targets = sum_targets_cache[s]
                loss_task = loss_task + bce_pos_neg(aT, ilp_targets.pos_idx, ilp_targets.neg_idx)

            loss_task = loss_task / B
            loss = loss_task + lam * loss_concepts

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        # Eval after epoch
        val_metrics = evaluate(
            cbm=cbm,
            learner=learner,
            loader=val_loader,
            device=device,
            n_atoms=n_atoms,
            bot_idx=bot_idx,
            T=bundle.program.T,
            soft_idx_digit=soft_idx_digit,
            hard_idx_add=hard_idx_add,
            sum_is_idx=sum_is_idx,
            sum_targets_cache=sum_targets_cache,
        )

        print(
            f"Epoch {ep+1}/{args.epochs} | "
            f"Train Loss: {total_loss/len(train_loader):.4f} | "
            f"Val Task Loss: {val_metrics['loss_task']:.4f} | "
            f"Val Concept Acc: {val_metrics['acc_concepts']:.3f} | "
            f"Val Sum Acc (ILP): {val_metrics['acc_sum_ilp']:.3f} | "
            f"Val Sum Acc (CBM): {val_metrics['acc_sum_cbm']:.3f}"
        )


if __name__ == "__main__":
    main()
