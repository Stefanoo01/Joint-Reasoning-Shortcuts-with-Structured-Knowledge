from __future__ import annotations

from experiments.run_halfmnist_supervised import is_better_validation
from experiments.run_halfmnist_supervised import resolve_seed_values


def test_resolve_seed_values_uses_default_base_seed():
    assert resolve_seed_values(None, 3) == [123, 124, 125]


def test_resolve_seed_values_preserves_explicit_zero_seed():
    assert resolve_seed_values(0, 2) == [0, 1]


def test_validation_selection_prefers_higher_ilp_accuracy():
    best = {"acc_sum_ilp": 0.91, "loss_task": 0.20, "acc_sum_cbm": 0.80}
    candidate = {"acc_sum_ilp": 0.92, "loss_task": 0.50, "acc_sum_cbm": 0.10}
    assert is_better_validation(candidate, best)


def test_validation_selection_uses_loss_then_cbm_as_tiebreakers():
    best = {"acc_sum_ilp": 0.92, "loss_task": 0.20, "acc_sum_cbm": 0.70}
    lower_loss = {"acc_sum_ilp": 0.92, "loss_task": 0.10, "acc_sum_cbm": 0.10}
    higher_cbm = {"acc_sum_ilp": 0.92, "loss_task": 0.20, "acc_sum_cbm": 0.75}
    worse_cbm = {"acc_sum_ilp": 0.92, "loss_task": 0.20, "acc_sum_cbm": 0.65}

    assert is_better_validation(lower_loss, best)
    assert is_better_validation(higher_cbm, best)
    assert not is_better_validation(worse_cbm, best)
