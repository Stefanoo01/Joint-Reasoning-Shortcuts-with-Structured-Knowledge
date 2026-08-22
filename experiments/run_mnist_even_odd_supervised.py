from __future__ import annotations

import argparse
import os
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

from configs.mnist_even_odd_addition import make_config
from configs.mnist_even_odd_presets import format_preset, get_preset, list_presets
from experiments.addition_evaluation import build_deterministic_loader
from experiments.addition_evaluation import clone_state_dict
from experiments.addition_evaluation import is_better_sum_validation
from experiments.addition_evaluation import print_multi_seed_summary
from experiments.addition_evaluation import resolve_seed_values
from experiments.addition_evaluation import set_global_seed
from ilp.learning.system_builder import build_system_from_config
from ilp.learning.trainer import extract_topk_program
from ilp.logic.atoms import Atom
from ilp.logic.valuation_soft import build_a0_from_indexed_facts


def build_add_truth_table_hard_idx(atom_to_idx) -> torch.Tensor:
    hard_atoms = []
    for a in range(10):
        for b in range(10):
            hard_atoms.append(Atom("add", (str(a), str(b), str(a + b))))
    return torch.tensor([atom_to_idx[a] for a in hard_atoms], dtype=torch.long)


def build_digit12_soft_idx(atom_to_idx) -> torch.Tensor:
    atoms = [Atom("digit1", (str(d),)) for d in range(10)]
    atoms.extend(Atom("digit2", (str(d),)) for d in range(10))
    return torch.tensor([atom_to_idx[a] for a in atoms], dtype=torch.long)


def build_sum_is_idx(atom_to_idx) -> torch.Tensor:
    atoms = [Atom("sum_is", (str(s),)) for s in range(19)]
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
    hard_idx_add: torch.Tensor,
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
            hard_idx=hard_idx_add,
        )
        outputs.append(learner.infer_T_paper(a0_chunk, T=T, temperature=1.0, fast=True))
    return torch.cat(outputs, dim=0)


def compute_sum_task_loss(
    scores: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    pos = scores.gather(1, targets.unsqueeze(1)).clamp(eps, 1 - eps)
    neg_mask = torch.ones_like(scores, dtype=torch.bool)
    neg_mask.scatter_(1, targets.unsqueeze(1), False)
    neg = scores[neg_mask].view(scores.size(0), -1).clamp(eps, 1 - eps)
    return -(pos.log()).mean() - ((1.0 - neg).log()).mean()


def print_available_presets(experiment: str) -> None:
    print(f"Available MNIST-Even-Odd presets for '{experiment}':")
    for preset in list_presets(experiment):
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
    ilp_chunk_size: int,
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

    from collections import Counter

    shortcut_map = Counter()

    for imgs, targets, concepts in loader:
        batch_size = imgs.size(0)
        imgs = imgs.to(device)
        d1_true_t = concepts[:, 0].long().to(device)
        d2_true_t = concepts[:, 1].long().to(device)
        s_true_t = targets.long().to(device)
        s_true_list = targets.long().tolist()

        cbm_out = cbm(imgs)
        logits1 = cbm_out["CS"][:, 0, :]
        logits2 = cbm_out["CS"][:, 1, :]
        probs1 = cbm_out["pCS"][:, 0, :]
        probs2 = cbm_out["pCS"][:, 1, :]

        d1_hat = torch.argmax(logits1, dim=1)
        d2_hat = torch.argmax(logits2, dim=1)
        correct_d1 += int(((d1_hat == d1_true_t) & (d1_true_t != -1)).sum().item())
        correct_d2 += int(((d2_hat == d2_true_t) & (d2_true_t != -1)).sum().item())

        for i, target_sum in enumerate(s_true_list):
            if target_sum in {6, 10, 12}:
                shortcut_map[(target_sum, int(d1_hat[i]), int(d2_hat[i]))] += 1

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
            hard_idx_add=hard_idx_add,
            ilp_chunk_size=ilp_chunk_size,
        )
        scores = aT_batch[:, sum_is_idx]

        correct_sum_cbm += int((d1_hat + d2_hat).eq(s_true_t).sum().item())
        s_hat_ilp = torch.argmax(scores, dim=1)
        correct_sum_ilp += int(s_hat_ilp.eq(s_true_t).sum().item())
        loss_task_sum += float(compute_sum_task_loss(scores, s_true_t).item()) * batch_size

        total += batch_size

    print("\n--- Model Reasoning Shortcut Mappings ---")
    for (s_true, d1, d2), count in sorted(shortcut_map.items()):
        if count > 10:
            print(
                f"When True Sum was {s_true}, CBM guessed digits: ({d1}, {d2}) [{count} times]"
            )
    print("------------------------------------------")

    return {
        "loss_task": loss_task_sum / total,
        "loss_concepts": loss_concepts_sum / total,
        "acc_concepts": ((correct_d1 + correct_d2) / (2 * total)) if total > 0 else 0.0,
        "acc_sum_cbm": correct_sum_cbm / total,
        "acc_sum_ilp": correct_sum_ilp / total,
    }


def run_single_seed(
    *,
    args,
    seed: int,
    dataset_cls,
    cbm_cls,
    print_program: bool,
) -> dict:
    seed_args = argparse.Namespace(**vars(args))
    seed_args.seed = seed
    set_global_seed(seed)
    device = torch.device("cpu")

    print(f"\n===== Seed {seed} =====")
    print("Loading RSBench SHORTMNIST (MNIST-Even-Odd)...")
    dataset = dataset_cls(seed_args)
    train_loader, _, _ = dataset.get_data_loaders()
    dataset.print_stats()

    val_loader = build_deterministic_loader(
        dataset.dataset_val,
        batch_size=seed_args.batch_size,
    )
    test_id_loader = build_deterministic_loader(
        dataset.dataset_test,
        batch_size=seed_args.batch_size,
    )
    ood_same_targets_loader = build_deterministic_loader(
        dataset.ood_test_2,
        batch_size=seed_args.batch_size,
    )
    ood_full_loader = build_deterministic_loader(
        dataset.ood_test,
        batch_size=seed_args.batch_size,
    )
    print(
        "Evaluation protocol | "
        f"selection<-validation ({len(dataset.dataset_val)}) | "
        f"test_id ({len(dataset.dataset_test)}) | "
        f"test_ood_same_sums ({len(dataset.ood_test_2)}) | "
        f"test_ood_full ({len(dataset.ood_test)})"
    )

    encoder, _ = dataset.get_backbone()
    cbm = cbm_cls(
        encoder=encoder,
        n_images=2,
        args=seed_args,
        n_facts=10,
        nr_classes=19,
    ).to(device)

    cfg = make_config(
        mode=seed_args.config_mode,
        T=seed_args.reasoning_steps,
        variant=seed_args.config_variant,
    )
    bundle = build_system_from_config(cfg)
    learner = bundle.learner.to(device)

    n_atoms = len(bundle.G)
    bot_idx = bundle.bot_idx
    atom_to_idx = bundle.atom_to_idx
    hard_idx_add = build_add_truth_table_hard_idx(atom_to_idx).to(device)
    soft_idx_digit = build_digit12_soft_idx(atom_to_idx).to(device)
    sum_is_idx = build_sum_is_idx(atom_to_idx).to(device)

    opt = torch.optim.Adam(list(cbm.parameters()) + list(learner.parameters()), lr=1e-3)

    def lambda_for_epoch(ep: int) -> float:
        if seed_args.lambda_mode == "fixed":
            return seed_args.lam2
        if ep == 0:
            return seed_args.lam0
        if ep == 1:
            return seed_args.lam1
        return seed_args.lam2

    best_epoch = 0
    best_val_metrics = None
    best_cbm_state = None
    best_learner_state = None

    evaluation_kwargs = {
        "cbm": cbm,
        "learner": learner,
        "device": device,
        "n_atoms": n_atoms,
        "bot_idx": bot_idx,
        "T": bundle.program.T,
        "soft_idx_digit": soft_idx_digit,
        "hard_idx_add": hard_idx_add,
        "sum_is_idx": sum_is_idx,
        "ilp_chunk_size": seed_args.ilp_chunk_size,
    }

    print("\nTraining...")
    for ep in range(seed_args.epochs):
        lam = lambda_for_epoch(ep)
        print(f"Epoch {ep + 1}/{seed_args.epochs} | Lambda: {lam}")
        cbm.train()
        learner.train()

        total_loss = 0.0
        for imgs, targets, concepts in train_loader:
            imgs = imgs.to(device)
            d1_true_t = concepts[:, 0].long().to(device)
            d2_true_t = concepts[:, 1].long().to(device)
            s_true_t = targets.long().to(device)

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
                hard_idx_add=hard_idx_add,
                ilp_chunk_size=seed_args.ilp_chunk_size,
            )
            scores = aT_batch[:, sum_is_idx]
            loss_task = compute_sum_task_loss(scores, s_true_t)
            loss = loss_task + lam * loss_concepts

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        val_metrics = evaluate(loader=val_loader, **evaluation_kwargs)
        if is_better_sum_validation(val_metrics, best_val_metrics):
            best_epoch = ep + 1
            best_val_metrics = dict(val_metrics)
            best_cbm_state = clone_state_dict(cbm)
            best_learner_state = clone_state_dict(learner)

        print(
            f"Epoch {ep + 1}/{seed_args.epochs} | "
            f"Train Loss: {total_loss / len(train_loader):.4f} | "
            f"Val Task Loss: {val_metrics['loss_task']:.4f} | "
            f"Val Concept Acc: {val_metrics['acc_concepts']:.3f} | "
            f"Val Sum Acc (ILP): {val_metrics['acc_sum_ilp']:.3f} | "
            f"Val Sum Acc (CBM): {val_metrics['acc_sum_cbm']:.3f}"
        )

    if best_cbm_state is None or best_learner_state is None:
        raise RuntimeError("No validation checkpoint was selected during training")

    cbm.load_state_dict(best_cbm_state)
    learner.load_state_dict(best_learner_state)
    split_loaders = {
        "val_metrics": val_loader,
        "test_id_metrics": test_id_loader,
        "ood_same_targets_metrics": ood_same_targets_loader,
        "ood_full_metrics": ood_full_loader,
    }
    metrics_by_split = {
        key: evaluate(loader=loader, **evaluation_kwargs)
        for key, loader in split_loaders.items()
    }

    print(f"\nBest-checkpoint evaluation | seed={seed} | epoch={best_epoch}")
    for split_name, key in (
        ("Validation", "val_metrics"),
        ("Test ID", "test_id_metrics"),
        ("Test OOD same sums", "ood_same_targets_metrics"),
        ("Test OOD full", "ood_full_metrics"),
    ):
        metrics = metrics_by_split[key]
        print(
            f"  {split_name:<19} | "
            f"concept_acc={metrics['acc_concepts']:.3f} | "
            f"sum_ilp={metrics['acc_sum_ilp']:.3f} | "
            f"sum_cbm={metrics['acc_sum_cbm']:.3f} | "
            f"task_loss={metrics['loss_task']:.4f}"
        )

    if print_program:
        print_learned_program(bundle)

    return {"seed": seed, "best_epoch": best_epoch, **metrics_by_split}


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--preset", type=str, default="add_medium_v1")
    bootstrap.add_argument("--list_presets", action="store_true")
    bootstrap_args, _ = bootstrap.parse_known_args()

    if bootstrap_args.list_presets:
        print_available_presets("addition")
        return

    preset = get_preset(bootstrap_args.preset)
    if preset.experiment != "addition":
        raise ValueError(
            f"Preset '{preset.name}' is for '{preset.experiment}', not 'addition'"
        )

    from datasets.shortcutmnist import SHORTMNIST
    from models.mnistcbm import MnistCBM, get_parser

    parser = get_parser()
    parser.add_argument("--preset", type=str, default=preset.name)
    parser.add_argument("--list_presets", action="store_true")
    parser.add_argument("--config_variant", type=str, default=preset.config_variant)
    parser.add_argument(
        "--config_mode", type=str, choices=["tight", "medium"], default=preset.config_mode
    )
    parser.add_argument("--reasoning_steps", type=int, default=preset.reasoning_steps)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--lambda_mode", type=str, choices=["fixed", "schedule"], default="fixed"
    )
    parser.add_argument("--lam0", type=float, default=1.0)
    parser.add_argument("--lam1", type=float, default=0.2)
    parser.add_argument("--lam2", type=float, default=0.0)
    parser.add_argument("--ilp_chunk_size", type=int, default=16)
    parser.add_argument("--num_seeds", type=int, default=1)

    parser.set_defaults(
        preset=preset.name,
        config_variant=preset.config_variant,
        config_mode=preset.config_mode,
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
        print_available_presets("addition")
        return

    selected_preset = get_preset(args.preset)
    if selected_preset.experiment != "addition":
        raise ValueError(
            f"Preset '{selected_preset.name}' is for '{selected_preset.experiment}', not 'addition'"
        )

    args.dataset = "shortmnist"
    args.task = "addition"
    if not hasattr(args, "c_sup") or args.c_sup == 0:
        args.c_sup = 1.0

    print("Using MNIST-Even-Odd preset:")
    print(" ", format_preset(selected_preset))
    print(
        "Resolved run settings | "
        f"variant={args.config_variant} | mode={args.config_mode} | T={args.reasoning_steps} | "
        f"epochs={args.epochs} | "
        f"batch_size={args.batch_size} | ilp_chunk_size={args.ilp_chunk_size} | "
        f"num_seeds={args.num_seeds} | base_seed={123 if args.seed is None else args.seed} | "
        f"lambda_mode={args.lambda_mode} | lam=({args.lam0}, {args.lam1}, {args.lam2})"
    )
    seed_values = resolve_seed_values(args.seed, args.num_seeds)
    results = [
        run_single_seed(
            args=args,
            seed=seed,
            dataset_cls=SHORTMNIST,
            cbm_cls=MnistCBM,
            print_program=args.num_seeds == 1,
        )
        for seed in seed_values
    ]
    print_multi_seed_summary(results, seed_values)


if __name__ == "__main__":
    main()
