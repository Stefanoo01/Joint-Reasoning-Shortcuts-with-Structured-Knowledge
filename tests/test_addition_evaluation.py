from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

import torch
from torch.utils.data import SequentialSampler

from experiments.addition_evaluation import build_deterministic_loader
from experiments.addition_evaluation import clone_state_dict
from experiments.addition_evaluation import is_better_sum_validation
from experiments.addition_evaluation import print_multi_seed_summary
from experiments.addition_evaluation import resolve_seed_values
from experiments.addition_evaluation import subset_with_targets
from experiments.addition_evaluation import summarize_values


class _TargetDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.targets = torch.tensor([0, 2, 5, 6, 8])

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return index, self.targets[index]


def test_target_subset_keeps_only_same_support_examples():
    dataset = _TargetDataset()
    subset = subset_with_targets(dataset, {0, 5, 6})

    assert subset.indices == [0, 2, 3]


def test_deterministic_loader_visits_each_example_once_in_order():
    dataset = _TargetDataset()
    loader = build_deterministic_loader(dataset, batch_size=2)

    assert isinstance(loader.sampler, SequentialSampler)
    assert list(loader.sampler) == list(range(len(dataset)))


def test_cloned_checkpoint_is_not_mutated_by_later_training():
    module = torch.nn.Linear(2, 1)
    checkpoint = clone_state_dict(module)
    saved_weight = checkpoint["weight"].clone()

    with torch.no_grad():
        module.weight.add_(1.0)

    assert torch.equal(checkpoint["weight"], saved_weight)
    assert not torch.equal(checkpoint["weight"], module.weight)


def test_validation_checkpoint_selection_prefers_ilp_then_loss_then_cbm():
    best = {"acc_sum_ilp": 0.90, "loss_task": 0.20, "acc_sum_cbm": 0.70}

    assert is_better_sum_validation(
        {"acc_sum_ilp": 0.91, "loss_task": 0.50, "acc_sum_cbm": 0.10},
        best,
    )
    assert is_better_sum_validation(
        {"acc_sum_ilp": 0.90, "loss_task": 0.10, "acc_sum_cbm": 0.10},
        best,
    )
    assert is_better_sum_validation(
        {"acc_sum_ilp": 0.90, "loss_task": 0.20, "acc_sum_cbm": 0.80},
        best,
    )
    assert not is_better_sum_validation(
        {"acc_sum_ilp": 0.89, "loss_task": 0.10, "acc_sum_cbm": 0.90},
        best,
    )


def test_seed_resolution_matches_sum_parity_protocol():
    assert resolve_seed_values(None, 3) == [123, 124, 125]
    assert resolve_seed_values(0, 2) == [0, 1]


def test_multi_seed_summary_uses_sample_standard_deviation():
    mean, std = summarize_values([0.7, 0.8, 0.9])

    assert abs(mean - 0.8) < 1e-12
    assert abs(std - 0.1) < 1e-12


def test_multi_seed_report_contains_every_evaluation_split():
    def metrics(value: float) -> dict:
        return {
            "acc_concepts": value,
            "acc_sum_ilp": value,
            "acc_sum_cbm": value,
            "loss_task": 1.0 - value,
        }

    results = [
        {
            "seed": seed,
            "best_epoch": index + 1,
            "val_metrics": metrics(value),
            "test_id_metrics": metrics(value),
            "ood_same_targets_metrics": metrics(value),
            "ood_full_metrics": metrics(value),
        }
        for index, (seed, value) in enumerate(((123, 0.7), (124, 0.9)))
    ]
    output = StringIO()
    with redirect_stdout(output):
        print_multi_seed_summary(results, [123, 124])

    report = output.getvalue()
    assert "seed=123" in report
    assert "seed=124" in report
    assert "Validation" in report
    assert "Test ID" in report
    assert "OOD same sums" in report
    assert "OOD full" in report
    assert "0.800±0.141" in report
