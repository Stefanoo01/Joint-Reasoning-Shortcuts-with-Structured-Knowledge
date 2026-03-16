from __future__ import annotations

from configs.half_mnist_addition import make_config as make_half_mnist_addition
from configs.half_mnist_peano import make_config as make_half_mnist_peano
from configs.half_mnist_presets import get_preset, list_presets
from configs.mnist_even_odd_addition import make_config as make_mnist_even_odd_addition
from configs.mnist_even_odd_presets import (
    get_preset as get_mnist_even_odd_preset,
)
from configs.mnist_even_odd_presets import (
    list_presets as list_mnist_even_odd_presets,
)
from configs.toy_even import make_config
from ilp.learning.system_builder import build_system_from_config


def test_build_system_from_config():
    cfg = make_config()
    bundle = build_system_from_config(cfg)

    assert len(bundle.G) > 0
    assert 0 <= bundle.bot_idx < len(bundle.idx_to_atom)
    assert bundle.idx_to_atom[bundle.bot_idx].pred == "__bot__"
    assert ("even", 1) in bundle.learner.caches
    assert ("succ2", 2) in bundle.learner.caches


def test_half_mnist_presets_drive_parametric_configs():
    add_preset = get_preset("add_tight_v1")
    add_cfg = make_half_mnist_addition(
        variant=add_preset.config_variant,
        mode=add_preset.config_mode,
        T=add_preset.reasoning_steps,
    )
    assert add_cfg.T == add_preset.reasoning_steps

    medium_preset = get_preset("add_medium_v1")
    medium_cfg = make_half_mnist_addition(
        variant=medium_preset.config_variant,
        mode=medium_preset.config_mode,
        T=medium_preset.reasoning_steps,
    )

    canonical_preset = get_preset("add_canonical_v1")
    canonical_cfg = make_half_mnist_addition(
        variant=canonical_preset.config_variant,
        mode=canonical_preset.config_mode,
        T=canonical_preset.reasoning_steps,
    )
    assert canonical_cfg.T == canonical_preset.reasoning_steps

    broad_preset = get_preset("add_broad_search_v1")
    broad_cfg = make_half_mnist_addition(
        variant=broad_preset.config_variant,
        mode=broad_preset.config_mode,
        T=broad_preset.reasoning_steps,
    )
    assert broad_cfg.T == broad_preset.reasoning_steps
    assert "digit1" in broad_cfg.bias.allowed_body_preds[("tmp", 2)]
    assert "add" in broad_cfg.bias.allowed_body_preds[("sum_is", 1)]
    base_bundle = build_system_from_config(medium_cfg)
    broad_bundle = build_system_from_config(broad_cfg)
    assert len(broad_bundle.clause_texts[("tmp", 2)][0]) > len(base_bundle.clause_texts[("tmp", 2)][0])
    assert len(broad_bundle.clause_texts[("sum_is", 1)][0]) > len(base_bundle.clause_texts[("sum_is", 1)][0])

    sum_relaxed_preset = get_preset("add_sum_relaxed_v1")
    sum_relaxed_cfg = make_half_mnist_addition(
        variant=sum_relaxed_preset.config_variant,
        mode=sum_relaxed_preset.config_mode,
        T=sum_relaxed_preset.reasoning_steps,
    )
    sum_relaxed_bundle = build_system_from_config(sum_relaxed_cfg)
    assert len(sum_relaxed_bundle.clause_texts[("tmp", 2)][0]) == len(base_bundle.clause_texts[("tmp", 2)][0])
    assert len(sum_relaxed_bundle.clause_texts[("sum_is", 1)][0]) > len(base_bundle.clause_texts[("sum_is", 1)][0])

    extra_tmp2_preset = get_preset("add_extra_tmp2_v1")
    extra_tmp2_cfg = make_half_mnist_addition(
        variant=extra_tmp2_preset.config_variant,
        mode=extra_tmp2_preset.config_mode,
        T=extra_tmp2_preset.reasoning_steps,
    )
    assert ("tmp2", 2) in extra_tmp2_cfg.aux_keys
    assert ("tmp2", 2) in extra_tmp2_cfg.templates
    assert "tmp2" in extra_tmp2_cfg.bias.allowed_body_preds[("sum_is", 1)]

    peano_preset = get_preset("peano_tight_memsafe_v1")
    peano_cfg = make_half_mnist_peano(
        variant=peano_preset.config_variant,
        mode=peano_preset.config_mode,
        T=peano_preset.reasoning_steps,
    )
    assert peano_cfg.T == peano_preset.reasoning_steps

    addition_names = {preset.name for preset in list_presets("addition")}
    peano_names = {preset.name for preset in list_presets("peano")}
    assert "add_medium_v1" in addition_names
    assert "add_canonical_v1" in addition_names
    assert "add_broad_search_v1" in addition_names
    assert "add_sum_relaxed_v1" in addition_names
    assert "add_extra_tmp2_v1" in addition_names
    assert "peano_medium_v1" in peano_names


def test_mnist_even_odd_presets_drive_parametric_configs():
    tight_preset = get_mnist_even_odd_preset("add_tight_v1")
    tight_cfg = make_mnist_even_odd_addition(
        variant=tight_preset.config_variant,
        mode=tight_preset.config_mode,
        T=tight_preset.reasoning_steps,
    )
    assert tight_cfg.T == tight_preset.reasoning_steps

    medium_preset = get_mnist_even_odd_preset("add_medium_v1")
    medium_cfg = make_mnist_even_odd_addition(
        variant=medium_preset.config_variant,
        mode=medium_preset.config_mode,
        T=medium_preset.reasoning_steps,
    )

    tight_bundle = build_system_from_config(tight_cfg)
    medium_bundle = build_system_from_config(medium_cfg)

    assert medium_cfg.T == medium_preset.reasoning_steps
    assert len(medium_bundle.clause_texts[("tmp", 2)][0]) > len(
        tight_bundle.clause_texts[("tmp", 2)][0]
    )
    assert len(medium_bundle.clause_texts[("sum_is", 1)][0]) == len(
        tight_bundle.clause_texts[("sum_is", 1)][0]
    )

    addition_names = {preset.name for preset in list_mnist_even_odd_presets("addition")}
    assert "add_medium_v1" in addition_names
    assert "add_tight_v1" in addition_names


if __name__ == "__main__":
    test_build_system_from_config()
    test_half_mnist_presets_drive_parametric_configs()
    test_mnist_even_odd_presets_drive_parametric_configs()
    print("✅ tests/test_task_config_build.py passed")
