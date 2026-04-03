from __future__ import annotations

import argparse
import sys
import os
import random
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 1. Mount RSBench in sys.path
RSBENCH_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "third_party", "rsbench-code", "rsseval", "rss"
))
if RSBENCH_DIR not in sys.path:
    sys.path.insert(0, RSBENCH_DIR)

# Differentiable ILP imports
from ilp.logic.atoms import Atom
from ilp.logic.valuation_soft import build_a0_from_indexed_facts
from configs.half_mnist_addition import make_config
from configs.half_mnist_presets import format_preset
from configs.half_mnist_presets import get_preset
from configs.half_mnist_presets import list_presets
from ilp.learning.system_builder import build_system_from_config
from ilp.learning.trainer import extract_topk_program


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

def build_sum_is_idx(atom_to_idx) -> torch.Tensor:
    # 0..8 to map directly index -> sum
    atoms = [Atom("sum_is", (str(s),)) for s in range(9)]
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


def compute_sum_task_loss(scores: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pos = scores.gather(1, targets.unsqueeze(1)).clamp(eps, 1 - eps)
    neg_mask = torch.ones_like(scores, dtype=torch.bool)
    neg_mask.scatter_(1, targets.unsqueeze(1), False)
    neg = scores[neg_mask].view(scores.size(0), -1).clamp(eps, 1 - eps)
    return -(pos.log()).mean() - ((1.0 - neg).log()).mean()


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


def print_available_presets(experiment: str) -> None:
    print(f"Available HalfMNIST presets for '{experiment}':")
    for preset in list_presets(experiment):
        print(" ", format_preset(preset))


def print_shortcut_mappings(shortcut_map, split_name: str | None = None, min_count: int = 10) -> None:
    label = split_name or "eval"
    print(f"\n--- Model Reasoning Shortcut Mappings ({label}) ---")
    for (s_true, d1, d2), count in sorted(shortcut_map.items()):
        if count > min_count:
            print(f"When True Sum was {s_true}, CBM guessed digits: ({d1}, {d2}) [{count} times]")
    print("--------------------------------------------------")


def resolve_seed_values(base_seed: int | None, num_seeds: int) -> list[int]:
    if num_seeds <= 0:
        raise ValueError("num_seeds must be > 0")
    seed0 = 123 if base_seed is None else base_seed
    return [seed0 + offset for offset in range(num_seeds)]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_project_loaders(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        dataset.dataset_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        dataset.dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    ood_loader = DataLoader(
        dataset.ood_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, ood_loader


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def is_better_validation(candidate: dict, best: dict | None) -> bool:
    if best is None:
        return True
    if candidate["acc_sum_ilp"] > best["acc_sum_ilp"]:
        return True
    if candidate["acc_sum_ilp"] < best["acc_sum_ilp"]:
        return False
    if candidate["loss_task"] < best["loss_task"]:
        return True
    if candidate["loss_task"] > best["loss_task"]:
        return False
    return candidate["acc_sum_cbm"] > best["acc_sum_cbm"]


def summarize_values(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


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
    split_name: str | None = None,
    print_mappings: bool = False,
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
        s_true_t = targets.long().to(device)
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

        # Track the mappings WRT the true sum
        for i in range(B):
            if s_true_list[i] in [0, 1, 5, 6]:
                shortcut_map[(s_true_list[i], int(d1_hat[i]), int(d2_hat[i]))] += 1

        loss_concepts = (F.cross_entropy(logits1, d1_true_t, ignore_index=-1) + F.cross_entropy(logits2, d2_true_t, ignore_index=-1)) / 2.0
        if torch.isnan(loss_concepts):
            loss_concepts = torch.tensor(0.0, device=device)
        loss_concepts_sum += float(loss_concepts.item()) * B

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
        loss_task_sum += float(compute_sum_task_loss(scores, s_true_t).item()) * B

        total += B


    if print_mappings:
        print_shortcut_mappings(shortcut_map, split_name=split_name)

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
    print("Loading RSBench HALFMNIST...")
    dataset = dataset_cls(seed_args)
    dataset.get_data_loaders()
    dataset.print_stats()

    train_loader, val_loader, ood_loader = build_project_loaders(
        dataset=dataset,
        batch_size=seed_args.batch_size,
        num_workers=seed_args.num_workers,
    )
    print(
        "Project protocol | "
        f"train<-original test ({len(dataset.dataset_test.data)}) | "
        f"selection<-validation ({len(dataset.dataset_val.data)}) | "
        f"final<-OOD ({len(dataset.ood_test.data)})"
    )

    encoder, _ = dataset.get_backbone()
    cbm = cbm_cls(
        encoder=encoder,
        n_images=2,
        args=seed_args,
        n_facts=5,
        nr_classes=9,
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

    opt = torch.optim.Adam(
        list(cbm.parameters()) + list(learner.parameters()),
        lr=seed_args.lr,
    )

    def lambda_for_epoch(ep: int) -> float:
        if seed_args.lambda_mode == "fixed":
            return seed_args.lam2
        if ep == 0:
            return seed_args.lam0
        if ep == 1:
            return seed_args.lam1
        return seed_args.lam2

    best_epoch = -1
    best_val_metrics = None
    best_cbm_state = None
    best_learner_state = None

    print("\nTraining...")
    for ep in range(seed_args.epochs):
        lam = lambda_for_epoch(ep)
        print(f"Epoch {ep+1}/{seed_args.epochs} | Lambda: {lam}")
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
            ilp_chunk_size=seed_args.ilp_chunk_size,
            split_name="val",
            print_mappings=False,
        )

        if is_better_validation(val_metrics, best_val_metrics):
            best_epoch = ep + 1
            best_val_metrics = val_metrics
            best_cbm_state = clone_state_dict(cbm)
            best_learner_state = clone_state_dict(learner)

        print(
            f"Epoch {ep+1}/{seed_args.epochs} | "
            f"Train Loss: {total_loss/len(train_loader):.4f} | "
            f"Val Task Loss: {val_metrics['loss_task']:.4f} | "
            f"Val Concept Acc: {val_metrics['acc_concepts']:.3f} | "
            f"Val Sum Acc (ILP): {val_metrics['acc_sum_ilp']:.3f} | "
            f"Val Sum Acc (CBM): {val_metrics['acc_sum_cbm']:.3f}"
        )

    if best_cbm_state is None or best_learner_state is None or best_val_metrics is None:
        raise RuntimeError("No validation checkpoint was selected during training")

    cbm.load_state_dict(best_cbm_state)
    learner.load_state_dict(best_learner_state)

    final_val_metrics = evaluate(
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
        ilp_chunk_size=seed_args.ilp_chunk_size,
        split_name="val",
        print_mappings=seed_args.print_mappings,
    )
    ood_metrics = evaluate(
        cbm=cbm,
        learner=learner,
        loader=ood_loader,
        device=device,
        n_atoms=n_atoms,
        bot_idx=bot_idx,
        T=bundle.program.T,
        soft_idx_digit=soft_idx_digit,
        hard_idx_add=hard_idx_add,
        sum_is_idx=sum_is_idx,
        ilp_chunk_size=seed_args.ilp_chunk_size,
        split_name="ood",
        print_mappings=seed_args.print_mappings,
    )

    print("\nBest-checkpoint evaluation")
    print(
        f"  Seed {seed} | best_epoch={best_epoch} | "
        f"val_concept_acc={final_val_metrics['acc_concepts']:.3f} | "
        f"val_sum_ilp={final_val_metrics['acc_sum_ilp']:.3f} | "
        f"val_sum_cbm={final_val_metrics['acc_sum_cbm']:.3f}"
    )
    print(
        f"  Seed {seed} | "
        f"ood_concept_acc={ood_metrics['acc_concepts']:.3f} | "
        f"ood_sum_ilp={ood_metrics['acc_sum_ilp']:.3f} | "
        f"ood_sum_cbm={ood_metrics['acc_sum_cbm']:.3f}"
    )

    if print_program:
        print_learned_program(bundle)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_metrics": final_val_metrics,
        "ood_metrics": ood_metrics,
    }


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
        raise ValueError(f"Preset '{preset.name}' is for '{preset.experiment}', not 'addition'")

    # RSBench imports
    from models.mnistcbm import get_parser
    from datasets.halfmnist import HALFMNIST
    from models.mnistcbm import MnistCBM

    parser = get_parser()
    parser.add_argument("--preset", type=str, default=preset.name)
    parser.add_argument("--list_presets", action="store_true")
    parser.add_argument("--config_variant", type=str, default=preset.config_variant)
    parser.add_argument("--config_mode", type=str, choices=["tight", "medium"], default=preset.config_mode)
    parser.add_argument("--reasoning_steps", type=int, default=preset.reasoning_steps)

    # Custom overrides for this specific script
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lambda_mode", type=str, choices=["fixed", "schedule"], default="fixed")
    parser.add_argument("--lam0", type=float, default=1.0)
    parser.add_argument("--lam1", type=float, default=0.2)
    parser.add_argument("--lam2", type=float, default=0.0)
    parser.add_argument("--ilp_chunk_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--print_mappings", action="store_true")

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

    # We will pretend the user passed reasonable defaults for RSBench internally
    args, _ = parser.parse_known_args()

    if args.list_presets:
        print_available_presets("addition")
        return

    selected_preset = get_preset(args.preset)
    if selected_preset.experiment != "addition":
        raise ValueError(f"Preset '{selected_preset.name}' is for '{selected_preset.experiment}', not 'addition'")
    
    # Force RSBench args to valid choices for HALFMNIST
    args.dataset = "halfmnist"
    args.task = "addition"
    if not hasattr(args, "c_sup") or args.c_sup == 0:
        args.c_sup = 1.0

    print("Using HalfMNIST preset:")
    print(" ", format_preset(selected_preset))
    print(
        "Resolved run settings | "
        f"variant={args.config_variant} | mode={args.config_mode} | T={args.reasoning_steps} | "
        f"epochs={args.epochs} | "
        f"batch_size={args.batch_size} | ilp_chunk_size={args.ilp_chunk_size} | "
        f"lr={args.lr} | num_workers={args.num_workers} | "
        f"num_seeds={args.num_seeds} | base_seed={123 if args.seed is None else args.seed} | "
        f"lambda_mode={args.lambda_mode} | lam=({args.lam0}, {args.lam1}, {args.lam2})"
    )
    seed_values = resolve_seed_values(args.seed, args.num_seeds)
    print("Project protocol | train on original test split, validate on validation split, final test on OOD split")

    results = []
    for seed in seed_values:
        results.append(
            run_single_seed(
                args=args,
                seed=seed,
                dataset_cls=HALFMNIST,
                cbm_cls=MnistCBM,
                print_program=args.num_seeds == 1,
            )
        )

    print("\nPer-seed summary")
    for result in results:
        print(
            f"  seed={result['seed']} | best_epoch={result['best_epoch']} | "
            f"val_sum_ilp={result['val_metrics']['acc_sum_ilp']:.3f} | "
            f"ood_sum_ilp={result['ood_metrics']['acc_sum_ilp']:.3f}"
        )

    if len(results) > 1:
        best_epoch_mean, best_epoch_std = summarize_values(
            [float(result["best_epoch"]) for result in results]
        )
        val_concepts_mean, val_concepts_std = summarize_values(
            [result["val_metrics"]["acc_concepts"] for result in results]
        )
        val_ilp_mean, val_ilp_std = summarize_values(
            [result["val_metrics"]["acc_sum_ilp"] for result in results]
        )
        val_cbm_mean, val_cbm_std = summarize_values(
            [result["val_metrics"]["acc_sum_cbm"] for result in results]
        )
        ood_concepts_mean, ood_concepts_std = summarize_values(
            [result["ood_metrics"]["acc_concepts"] for result in results]
        )
        ood_ilp_mean, ood_ilp_std = summarize_values(
            [result["ood_metrics"]["acc_sum_ilp"] for result in results]
        )
        ood_cbm_mean, ood_cbm_std = summarize_values(
            [result["ood_metrics"]["acc_sum_cbm"] for result in results]
        )

        print("\nAggregate over seeds")
        print(f"  Seeds      | {seed_values}")
        print(f"  Best epoch | mean={best_epoch_mean:.2f} | std={best_epoch_std:.2f}")
        print(
            f"  Val        | concept_acc={val_concepts_mean:.3f}±{val_concepts_std:.3f} | "
            f"sum_ilp={val_ilp_mean:.3f}±{val_ilp_std:.3f} | "
            f"sum_cbm={val_cbm_mean:.3f}±{val_cbm_std:.3f}"
        )
        print(
            f"  Test OOD   | concept_acc={ood_concepts_mean:.3f}±{ood_concepts_std:.3f} | "
            f"sum_ilp={ood_ilp_mean:.3f}±{ood_ilp_std:.3f} | "
            f"sum_cbm={ood_cbm_mean:.3f}±{ood_cbm_std:.3f}"
        )


if __name__ == "__main__":
    main()
