from __future__ import annotations

import os
import sys

import numpy as np

from configs.mnist_sum_parity import make_config
from configs.mnist_sum_parity_presets import get_preset, list_presets
from ilp.learning.system_builder import build_system_from_config


RSBENCH_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "third_party",
        "rsbench-code",
        "rsseval",
        "rss",
    )
)
if RSBENCH_DIR not in sys.path:
    sys.path.insert(0, RSBENCH_DIR)

from sumparity_split import in_distribution_mask, ood_mask, sum_parity_labels


def test_mnist_sum_parity_presets_drive_parametric_configs():
    tight_preset = get_preset("biased_tight_v1")
    tight_cfg = make_config(
        variant=tight_preset.config_variant,
        mode=tight_preset.config_mode,
        T=tight_preset.reasoning_steps,
    )
    medium_preset = get_preset("biased_medium_v1")
    medium_cfg = make_config(
        variant=medium_preset.config_variant,
        mode=medium_preset.config_mode,
        T=medium_preset.reasoning_steps,
    )
    tight_v2_preset = get_preset("biased_tight_v2")
    medium_v2_preset = get_preset("biased_medium_v2")

    tight_bundle = build_system_from_config(tight_cfg)
    medium_bundle = build_system_from_config(medium_cfg)

    assert tight_cfg.target_key == ("sum_parity", 1)
    assert ("sum_is", 1) in tight_cfg.aux_keys
    assert ("tmp", 2) in tight_cfg.aux_keys
    assert tight_preset.experiment == "sum_parity"
    assert tight_cfg.T == tight_preset.reasoning_steps
    assert medium_cfg.T == medium_preset.reasoning_steps
    assert (tight_preset.lam0, tight_preset.lam1, tight_preset.lam2) == (1.0, 0.2, 0.0)
    assert (medium_preset.lam0, medium_preset.lam1, medium_preset.lam2) == (1.0, 0.2, 0.0)
    assert tight_v2_preset.batch_size > tight_preset.batch_size
    assert tight_v2_preset.ilp_chunk_size > tight_preset.ilp_chunk_size
    assert medium_v2_preset.batch_size > medium_preset.batch_size
    assert medium_v2_preset.ilp_chunk_size > medium_preset.ilp_chunk_size
    assert len(medium_bundle.clause_texts[("tmp", 2)][0]) > len(
        tight_bundle.clause_texts[("tmp", 2)][0]
    )

    preset_names = {preset.name for preset in list_presets("sum_parity")}
    assert "biased_tight_v1" in preset_names
    assert "biased_tight_v2" in preset_names
    assert "biased_medium_v1" in preset_names
    assert "biased_medium_v2" in preset_names


def test_sum_parity_labels_and_masks_match_biased_split():
    labels = sum_parity_labels(
        np.array([0, 1, 2, 3]),
        np.array([0, 0, 3, 4]),
    )
    assert labels.tolist() == [0, 1, 1, 1]

    concepts = np.array(
        [
            [0, 0],  # even, even -> ID
            [1, 1],  # odd, odd -> ID
            [1, 2],  # odd, even -> ID
            [2, 1],  # even, odd -> OOD
            [4, 8],  # even, even -> ID
            [3, 9],  # odd, odd -> ID
        ]
    )

    assert in_distribution_mask(concepts).tolist() == [True, True, True, False, True, True]
    assert ood_mask(concepts).tolist() == [False, False, False, True, False, False]
