from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

RSBENCH_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "third_party", "rsbench-code", "rsseval", "rss"
    )
)
if RSBENCH_DIR not in sys.path:
    sys.path.insert(0, RSBENCH_DIR)

from configs.mnist_sum_parity import make_config
from configs.mnist_sum_parity_presets import format_preset, get_preset, list_presets
from ilp.learning.system_builder import build_system_from_config
from ilp.learning.trainer import extract_topk_program
from ilp.logic.atoms import Atom
from ilp.logic.valuation_soft import build_a0_from_indexed_facts


def build_hard_idx(atom_to_idx, n_digits: int) -> torch.Tensor:
    hard_atoms = []
    for a in range(n_digits):
        for b in range(n_digits):
            hard_atoms.append(Atom("add", (str(a), str(b), str(a + b))))
    for s in range((2 * n_digits) - 1):
        parity = "odd" if s % 2 else "even"
        hard_atoms.append(Atom("parity_of", (str(s), parity)))
    return torch.tensor([atom_to_idx[a] for a in hard_atoms], dtype=torch.long)


def build_digit12_soft_idx(atom_to_idx, n_digits: int) -> torch.Tensor:
    atoms = [Atom("digit1", (str(d),)) for d in range(n_digits)]
    atoms.extend(Atom("digit2", (str(d),)) for d in range(n_digits))
    return torch.tensor([atom_to_idx[a] for a in atoms], dtype=torch.long)


def build_sum_parity_idx(atom_to_idx) -> torch.Tensor:
    atoms = [Atom("sum_parity", ("even",)), Atom("sum_parity", ("odd",))]
    return torch.tensor([atom_to_idx[a] for a in atoms], dtype=torch.long)


def infer_ilp_in_chunks(
    *,
    learner: nn.Module,
    probs1: torch.Tensor,
    probs2: torch.Tensor,
    n_atoms: int,
    bot_idx: int,
    T: int,
    soft_idx_digit: torch.Tensor,
    hard_idx: torch.Tensor,
    ilp_chunk_size: int,
) -> torch.Tensor:
    if ilp_chunk_size <= 0:
        raise ValueError("ilp_chunk_size must be > 0")
    soft_vals = torch.cat([probs1, probs2], dim=1)
    outputs = []
    for soft_chunk in soft_vals.split(ilp_chunk_size, dim=0):
        a0_chunk = build_a0_from_indexed_facts(
            n=n_atoms,
            bot_idx=bot_idx,
            soft_idx=soft_idx_digit,
            soft_val=soft_chunk,
            hard_idx=hard_idx,
        )
        outputs.append(learner.infer_T_paper(a0_chunk, T=T, temperature=1.0, fast=True))
    return torch.cat(outputs, dim=0)


def compute_task_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if scores.size(1) != 2:
        raise ValueError("compute_task_loss currently expects exactly 2 parity classes")

    pos = scores.gather(1, targets.unsqueeze(1)).clamp(eps, 1 - eps)
    neg_mask = torch.ones_like(scores, dtype=torch.bool)
    neg_mask.scatter_(1, targets.unsqueeze(1), False)
    neg = scores[neg_mask].view(scores.size(0), -1).clamp(eps, 1 - eps)

    if class_weights is None:
        pos_weights = torch.ones_like(pos)
        neg_weights = torch.ones_like(neg)
    else:
        pos_weights = class_weights[targets].unsqueeze(1)
        neg_targets = 1 - targets
        neg_weights = class_weights[neg_targets].unsqueeze(1)

    pos_loss = -(pos_weights * pos.log()).mean()
    neg_loss = -(neg_weights * (1.0 - neg).log()).mean()
    return pos_loss + neg_loss


def build_parity_class_weights(targets, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(torch.as_tensor(targets, dtype=torch.long), minlength=2).float()
    weights = counts.sum() / (counts.clamp_min(1.0) * counts.numel())
    return weights.to(device)


def print_available_presets() -> None:
    print("Available MNIST-SumParity presets:")
    for preset in list_presets("sum_parity"):
        print(" ", format_preset(preset))


def print_learned_program(bundle, temperature: float = 0.2, top_k: int = 5) -> None:
    ranked = extract_topk_program(bundle.learner, k=top_k, temperature=temperature)
    print(f"\n=== TOP-{top_k} ILP CLAUSE PAIRS ===")
    for key, entries in ranked.items():
        print(key)
        if key not in bundle.clause_texts:
            continue
        c1_texts, c2_texts = bundle.clause_texts[key]
        for rank, (j, k, prob) in enumerate(entries, start=1):
            print(f"  #{rank}: (j={j}, k={k}) prob={prob:.3f}")
            print("    C1:", c1_texts[j])
            print("    C2:", c2_texts[k])


def print_digit_pair_mappings(
    *,
    pair_counts,
    pair_totals,
    split_name: str,
    max_rows: int = 40,
) -> None:
    if not pair_totals:
        return

    print(f"\n--- Learned Digit Pair Mappings ({split_name}) ---")
    rows_printed = 0
    for true_pair in sorted(pair_totals):
        total = pair_totals[true_pair]
        ranked = sorted(
            (
                (pred_pair, count)
                for (tp, pred_pair), count in pair_counts.items()
                if tp == true_pair
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if not ranked:
            continue

        pred_pair, count = ranked[0]
        ratio = count / total
        print(
            f"True digits {true_pair} -> Pred digits {pred_pair} "
            f"[{count}/{total} | {ratio:.1%}]"
        )
        rows_printed += 1
        if rows_printed >= max_rows:
            remaining = len(pair_totals) - rows_printed
            if remaining > 0:
                print(f"... {remaining} more true-digit mappings omitted")
            break
    print("---------------------------------------------")


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
    hard_idx: torch.Tensor,
    sum_parity_idx: torch.Tensor,
    ilp_chunk_size: int,
    class_weights: torch.Tensor,
    split_name: str | None = None,
    print_mappings: bool = False,
) -> dict:
    cbm.eval()
    learner.eval()

    total = 0
    correct_d1 = 0
    correct_d2 = 0
    correct_parity_cbm = 0
    correct_parity_ilp = 0
    loss_task_sum = 0.0
    loss_concepts_sum = 0.0

    from collections import Counter

    pair_counts = Counter()
    pair_totals = Counter()

    for imgs, targets, concepts in loader:
        batch_size = imgs.size(0)
        imgs = imgs.to(device)
        d1_true_t = concepts[:, 0].long().to(device)
        d2_true_t = concepts[:, 1].long().to(device)
        y_true_t = targets.long().to(device)

        cbm_out = cbm(imgs)
        logits1 = cbm_out["CS"][:, 0, :]
        logits2 = cbm_out["CS"][:, 1, :]
        probs1 = cbm_out["pCS"][:, 0, :]
        probs2 = cbm_out["pCS"][:, 1, :]

        d1_hat = torch.argmax(logits1, dim=1)
        d2_hat = torch.argmax(logits2, dim=1)
        correct_d1 += int(((d1_hat == d1_true_t) & (d1_true_t != -1)).sum().item())
        correct_d2 += int(((d2_hat == d2_true_t) & (d2_true_t != -1)).sum().item())

        for i in range(batch_size):
            true_pair = (int(d1_true_t[i]), int(d2_true_t[i]))
            pred_pair = (int(d1_hat[i]), int(d2_hat[i]))
            pair_counts[(true_pair, pred_pair)] += 1
            pair_totals[true_pair] += 1

        loss_concepts = (
            F.cross_entropy(logits1, d1_true_t, ignore_index=-1)
            + F.cross_entropy(logits2, d2_true_t, ignore_index=-1)
        ) / 2.0
        if torch.isnan(loss_concepts):
            loss_concepts = torch.tensor(0.0, device=device)
        loss_concepts_sum += float(loss_concepts.item()) * batch_size

        aT_batch = infer_ilp_in_chunks(
            learner=learner,
            probs1=probs1,
            probs2=probs2,
            n_atoms=n_atoms,
            bot_idx=bot_idx,
            T=T,
            soft_idx_digit=soft_idx_digit,
            hard_idx=hard_idx,
            ilp_chunk_size=ilp_chunk_size,
        )
        scores = aT_batch[:, sum_parity_idx]

        parity_hat_cbm = ((d1_hat + d2_hat) % 2).long()
        correct_parity_cbm += int(parity_hat_cbm.eq(y_true_t).sum().item())

        parity_hat_ilp = torch.argmax(scores, dim=1)
        correct_parity_ilp += int(parity_hat_ilp.eq(y_true_t).sum().item())
        loss_task_sum += float(
            compute_task_loss(scores, y_true_t, class_weights=class_weights).item()
        ) * batch_size

        total += batch_size

    if print_mappings:
        print_digit_pair_mappings(
            pair_counts=pair_counts,
            pair_totals=pair_totals,
            split_name=split_name or "eval",
        )

    return {
        "loss_task": loss_task_sum / total,
        "loss_concepts": loss_concepts_sum / total,
        "acc_concepts": ((correct_d1 + correct_d2) / (2 * total)) if total > 0 else 0.0,
        "acc_parity_cbm": correct_parity_cbm / total,
        "acc_parity_ilp": correct_parity_ilp / total,
    }


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--preset", type=str, default="biased_tight_v1")
    bootstrap.add_argument("--list_presets", action="store_true")
    bootstrap_args, _ = bootstrap.parse_known_args()

    if bootstrap_args.list_presets:
        print_available_presets()
        return

    preset = get_preset(bootstrap_args.preset)
    if preset.experiment != "sum_parity":
        raise ValueError(
            f"Preset '{preset.name}' is for '{preset.experiment}', not 'sum_parity'"
        )

    from datasets.sumparitymnist import SUMPARITYMNIST
    from models.mnistsumparitycbm import MnistSumParityCBM, get_parser

    parser = get_parser()
    parser.add_argument("--preset", type=str, default=preset.name)
    parser.add_argument("--list_presets", action="store_true")
    parser.add_argument("--config_variant", type=str, default=preset.config_variant)
    parser.add_argument(
        "--config_mode", type=str, choices=["tight", "medium"], default=preset.config_mode
    )
    parser.add_argument("--n_digits", type=int, default=preset.n_digits)
    parser.add_argument("--reasoning_steps", type=int, default=preset.reasoning_steps)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--balanced_sampler", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--lambda_mode", type=str, choices=["fixed", "schedule"], default="fixed"
    )
    parser.add_argument("--lam0", type=float, default=1.0)
    parser.add_argument("--lam1", type=float, default=0.2)
    parser.add_argument("--lam2", type=float, default=0.0)
    parser.add_argument("--ilp_chunk_size", type=int, default=16)

    parser.set_defaults(
        preset=preset.name,
        config_variant=preset.config_variant,
        config_mode=preset.config_mode,
        n_digits=preset.n_digits,
        reasoning_steps=preset.reasoning_steps,
        epochs=preset.epochs,
        batch_size=preset.batch_size,
        ilp_chunk_size=preset.ilp_chunk_size,
        lambda_mode=preset.lambda_mode,
        lam0=preset.lam0,
        lam1=preset.lam1,
        lam2=preset.lam2,
    )

    args, _ = parser.parse_known_args()

    if args.list_presets:
        print_available_presets()
        return

    selected_preset = get_preset(args.preset)
    if selected_preset.experiment != "sum_parity":
        raise ValueError(
            f"Preset '{selected_preset.name}' is for '{selected_preset.experiment}', not 'sum_parity'"
        )

    args.dataset = "sumparitymnist"
    args.task = "sum_parity"
    if not hasattr(args, "c_sup") or args.c_sup == 0:
        args.c_sup = 1.0

    print("Using MNIST-SumParity preset:")
    print(" ", format_preset(selected_preset))
    print(
        "Resolved run settings | "
        f"variant={args.config_variant} | mode={args.config_mode} | n_digits={args.n_digits} | "
        f"T={args.reasoning_steps} | "
        f"epochs={args.epochs} | "
        f"batch_size={args.batch_size} | ilp_chunk_size={args.ilp_chunk_size} | "
        f"lr={args.lr} | balanced_sampler={args.balanced_sampler} | num_workers={args.num_workers} | "
        f"lambda_mode={args.lambda_mode} | lam=({args.lam0}, {args.lam1}, {args.lam2})"
    )

    random.seed(args.seed or 123)
    torch.manual_seed(args.seed or 123)
    device = torch.device("cpu")

    print("Loading biased MNIST-SumParity dataset...")
    dataset = SUMPARITYMNIST(args)
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    dataset.print_stats()

    encoder, _ = dataset.get_backbone()
    cbm = MnistSumParityCBM(
        encoder=encoder,
        n_images=2,
        args=args,
        n_facts=args.n_digits,
        nr_classes=2,
    ).to(device)

    cfg = make_config(
        mode=args.config_mode,
        T=args.reasoning_steps,
        variant=args.config_variant,
        n_digits=args.n_digits,
    )
    bundle = build_system_from_config(cfg)
    learner = bundle.learner.to(device)

    n_atoms = len(bundle.G)
    bot_idx = bundle.bot_idx
    atom_to_idx = bundle.atom_to_idx

    hard_idx = build_hard_idx(atom_to_idx, args.n_digits).to(device)
    soft_idx_digit = build_digit12_soft_idx(atom_to_idx, args.n_digits).to(device)
    sum_parity_idx = build_sum_parity_idx(atom_to_idx).to(device)
    class_weights = build_parity_class_weights(dataset.dataset_train.targets, device)

    opt = torch.optim.Adam(
        list(cbm.parameters()) + list(learner.parameters()),
        lr=args.lr,
    )

    def lambda_for_epoch(ep: int) -> float:
        if args.lambda_mode == "fixed":
            return args.lam2
        if ep == 0:
            return args.lam0
        if ep == 1:
            return args.lam1
        return args.lam2

    print("\nTraining...")
    for ep in range(args.epochs):
        lam = lambda_for_epoch(ep)
        print(f"Epoch {ep + 1}/{args.epochs} | Lambda: {lam}")
        cbm.train()
        learner.train()

        total_loss = 0.0
        for imgs, targets, concepts in train_loader:
            imgs = imgs.to(device)
            d1_true_t = concepts[:, 0].long().to(device)
            d2_true_t = concepts[:, 1].long().to(device)
            y_true_t = targets.long().to(device)

            cbm_out = cbm(imgs)
            logits1 = cbm_out["CS"][:, 0, :]
            logits2 = cbm_out["CS"][:, 1, :]
            probs1 = cbm_out["pCS"][:, 0, :]
            probs2 = cbm_out["pCS"][:, 1, :]

            loss_concepts = (
                F.cross_entropy(logits1, d1_true_t, ignore_index=-1)
                + F.cross_entropy(logits2, d2_true_t, ignore_index=-1)
            ) / 2.0
            if torch.isnan(loss_concepts):
                loss_concepts = torch.tensor(0.0, device=device)

            aT_batch = infer_ilp_in_chunks(
                learner=learner,
                probs1=probs1,
                probs2=probs2,
                n_atoms=n_atoms,
                bot_idx=bot_idx,
                T=bundle.program.T,
                soft_idx_digit=soft_idx_digit,
                hard_idx=hard_idx,
                ilp_chunk_size=args.ilp_chunk_size,
            )
            scores = aT_batch[:, sum_parity_idx]
            loss_task = compute_task_loss(scores, y_true_t, class_weights=class_weights)
            loss = loss_task + lam * loss_concepts

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        val_metrics = evaluate(
            cbm=cbm,
            learner=learner,
            loader=val_loader,
            device=device,
            n_atoms=n_atoms,
            bot_idx=bot_idx,
            T=bundle.program.T,
            soft_idx_digit=soft_idx_digit,
            hard_idx=hard_idx,
            sum_parity_idx=sum_parity_idx,
            ilp_chunk_size=args.ilp_chunk_size,
            class_weights=class_weights,
            split_name="val",
            print_mappings=True,
        )

        print(
            f"Epoch {ep + 1}/{args.epochs} | "
            f"Train Loss: {total_loss / len(train_loader):.4f} | "
            f"Val Task Loss: {val_metrics['loss_task']:.4f} | "
            f"Val Concept Acc: {val_metrics['acc_concepts']:.3f} | "
            f"Val Parity Acc (ILP): {val_metrics['acc_parity_ilp']:.3f} | "
            f"Val Parity Acc (CBM): {val_metrics['acc_parity_cbm']:.3f}"
        )

    test_metrics = evaluate(
        cbm=cbm,
        learner=learner,
        loader=test_loader,
        device=device,
        n_atoms=n_atoms,
        bot_idx=bot_idx,
        T=bundle.program.T,
        soft_idx_digit=soft_idx_digit,
        hard_idx=hard_idx,
        sum_parity_idx=sum_parity_idx,
        ilp_chunk_size=args.ilp_chunk_size,
        class_weights=class_weights,
        print_mappings=True,
    )
    ood_metrics = evaluate(
        cbm=cbm,
        learner=learner,
        loader=dataset.ood_loader,
        device=device,
        n_atoms=n_atoms,
        bot_idx=bot_idx,
        T=bundle.program.T,
        soft_idx_digit=soft_idx_digit,
        hard_idx=hard_idx,
        sum_parity_idx=sum_parity_idx,
        ilp_chunk_size=args.ilp_chunk_size,
        class_weights=class_weights,
        print_mappings=True,
    )

    print("\nFinal evaluation")
    print(
        f"  Test ID   | concept_acc={test_metrics['acc_concepts']:.3f} | "
        f"parity_ilp={test_metrics['acc_parity_ilp']:.3f} | parity_cbm={test_metrics['acc_parity_cbm']:.3f}"
    )
    print(
        f"  Test OOD  | concept_acc={ood_metrics['acc_concepts']:.3f} | "
        f"parity_ilp={ood_metrics['acc_parity_ilp']:.3f} | parity_cbm={ood_metrics['acc_parity_cbm']:.3f}"
    )

    print_learned_program(bundle)


if __name__ == "__main__":
    main()
