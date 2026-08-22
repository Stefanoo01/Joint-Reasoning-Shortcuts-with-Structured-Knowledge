from __future__ import annotations

from collections.abc import Iterable
import random
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


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


def summarize_values(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def print_multi_seed_summary(results: list[dict], seed_values: list[int]) -> None:
    print("\nPer-seed summary")
    for result in results:
        print(
            f"  seed={result['seed']} | best_epoch={result['best_epoch']} | "
            f"val_sum_ilp={result['val_metrics']['acc_sum_ilp']:.3f} | "
            f"test_id_sum_ilp={result['test_id_metrics']['acc_sum_ilp']:.3f} | "
            f"ood_same_sum_ilp={result['ood_same_targets_metrics']['acc_sum_ilp']:.3f} | "
            f"ood_full_sum_ilp={result['ood_full_metrics']['acc_sum_ilp']:.3f}"
        )

    if len(results) == 1:
        return

    best_epoch_mean, best_epoch_std = summarize_values(
        [float(result["best_epoch"]) for result in results]
    )
    print("\nAggregate over seeds")
    print(f"  Seeds      | {seed_values}")
    print(f"  Best epoch | mean={best_epoch_mean:.2f} | std={best_epoch_std:.2f}")
    for split_name, key in (
        ("Validation", "val_metrics"),
        ("Test ID", "test_id_metrics"),
        ("OOD same sums", "ood_same_targets_metrics"),
        ("OOD full", "ood_full_metrics"),
    ):
        concept_mean, concept_std = summarize_values(
            [result[key]["acc_concepts"] for result in results]
        )
        ilp_mean, ilp_std = summarize_values(
            [result[key]["acc_sum_ilp"] for result in results]
        )
        cbm_mean, cbm_std = summarize_values(
            [result[key]["acc_sum_cbm"] for result in results]
        )
        print(
            f"  {split_name:<13} | "
            f"concept_acc={concept_mean:.3f}±{concept_std:.3f} | "
            f"sum_ilp={ilp_mean:.3f}±{ilp_std:.3f} | "
            f"sum_cbm={cbm_mean:.3f}±{cbm_std:.3f}"
        )


def build_deterministic_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Build an evaluation loader that visits every example exactly once."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


def subset_with_targets(dataset: Dataset, allowed_targets: Iterable[int]) -> Subset:
    """Select examples whose task target is in ``allowed_targets``."""
    allowed = {int(target) for target in allowed_targets}
    targets = torch.as_tensor(dataset.targets).reshape(-1)
    indices = [
        index
        for index, target in enumerate(targets.tolist())
        if int(target) in allowed
    ]
    if not indices:
        raise ValueError(f"No examples found for targets {sorted(allowed)}")
    return Subset(dataset, indices)


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Copy a module state to CPU so later training cannot mutate it."""
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def is_better_sum_validation(candidate: dict, best: dict | None) -> bool:
    """Rank checkpoints using validation data only."""
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
