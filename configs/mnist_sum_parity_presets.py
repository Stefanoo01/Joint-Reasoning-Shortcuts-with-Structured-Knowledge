from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal


MnistSumParityExperiment = Literal["sum_parity"]
MnistSumParityMode = Literal["tight", "medium"]
LambdaMode = Literal["fixed", "schedule"]


@dataclass(frozen=True)
class MnistSumParityPreset:
    name: str
    experiment: MnistSumParityExperiment
    config_variant: str
    config_mode: MnistSumParityMode
    n_digits: int
    reasoning_steps: int
    epochs: int
    batch_size: int
    ilp_chunk_size: int
    lambda_mode: LambdaMode
    lam0: float
    lam1: float
    lam2: float
    notes: str = ""


MNIST_SUM_PARITY_PRESETS: Dict[str, MnistSumParityPreset] = {
    "biased_tight_v1": MnistSumParityPreset(
        name="biased_tight_v1",
        experiment="sum_parity",
        config_variant="base",
        config_mode="tight",
        n_digits=10,
        reasoning_steps=3,
        epochs=30,
        batch_size=64,
        ilp_chunk_size=16,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Biased MNIST-SumParity with a tight symbolic bias forcing digit -> sum -> parity.",
    ),
    "biased_tight_v2": MnistSumParityPreset(
        name="biased_tight_v2",
        experiment="sum_parity",
        config_variant="base",
        config_mode="tight",
        n_digits=10,
        reasoning_steps=3,
        epochs=30,
        batch_size=128,
        ilp_chunk_size=32,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Tight biased MNIST-SumParity preset with larger batch and ILP chunks for faster GPU throughput.",
    ),
    "biased_medium_v1": MnistSumParityPreset(
        name="biased_medium_v1",
        experiment="sum_parity",
        config_variant="base",
        config_mode="medium",
        n_digits=10,
        reasoning_steps=3,
        epochs=30,
        batch_size=64,
        ilp_chunk_size=16,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Same biased task with a slightly wider clause search on the add step.",
    ),
    "biased_medium_v2": MnistSumParityPreset(
        name="biased_medium_v2",
        experiment="sum_parity",
        config_variant="base",
        config_mode="medium",
        n_digits=10,
        reasoning_steps=3,
        epochs=30,
        batch_size=128,
        ilp_chunk_size=32,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Medium biased MNIST-SumParity preset with larger batch and ILP chunks for faster GPU throughput.",
    ),
    "biased_broad_search_v1": MnistSumParityPreset(
        name="biased_broad_search_v1",
        experiment="sum_parity",
        config_variant="broad_search",
        config_mode="medium",
        n_digits=10,
        reasoning_steps=3,
        epochs=30,
        batch_size=64,
        ilp_chunk_size=16,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Broader MNIST-SumParity search space opening tmp, sum_is, and sum_parity while keeping clause shapes structured.",
    ),
    "biased_tight_0to5_v1": MnistSumParityPreset(
        name="biased_tight_0to5_v1",
        experiment="sum_parity",
        config_variant="base",
        config_mode="tight",
        n_digits=6,
        reasoning_steps=3,
        epochs=30,
        batch_size=32,
        ilp_chunk_size=32,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Reduced biased MNIST-SumParity over digits 0..5 with larger GPU-friendly batch and ILP chunks.",
    ),
    "biased_medium_0to5_v1": MnistSumParityPreset(
        name="biased_medium_0to5_v1",
        experiment="sum_parity",
        config_variant="base",
        config_mode="medium",
        n_digits=6,
        reasoning_steps=3,
        epochs=30,
        batch_size=32,
        ilp_chunk_size=32,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Reduced biased MNIST-SumParity over digits 0..5 with the wider medium clause search.",
    ),
    "biased_tmp_broad_only_0to5_v1": MnistSumParityPreset(
        name="biased_tmp_broad_only_0to5_v1",
        experiment="sum_parity",
        config_variant="tmp_broad_only",
        config_mode="medium",
        n_digits=6,
        reasoning_steps=3,
        epochs=30,
        batch_size=64,
        ilp_chunk_size=32,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Reduced 0..5 MNIST-SumParity with a wider search only on tmp while keeping sum_is and sum_parity guided.",
    ),
    "biased_broad_search_0to5_v1": MnistSumParityPreset(
        name="biased_broad_search_0to5_v1",
        experiment="sum_parity",
        config_variant="broad_search",
        config_mode="medium",
        n_digits=6,
        reasoning_steps=3,
        epochs=30,
        batch_size=64,
        ilp_chunk_size=32,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Reduced 0..5 MNIST-SumParity with a substantially wider but still structured symbolic search space.",
    ),
}


def get_preset(name: str) -> MnistSumParityPreset:
    try:
        return MNIST_SUM_PARITY_PRESETS[name]
    except KeyError as exc:
        known = ", ".join(MNIST_SUM_PARITY_PRESETS.keys())
        raise KeyError(
            f"Unknown MNIST-SumParity preset '{name}'. Known presets: {known}"
        ) from exc


def list_presets(
    experiment: MnistSumParityExperiment | None = None,
) -> List[MnistSumParityPreset]:
    presets = list(MNIST_SUM_PARITY_PRESETS.values())
    if experiment is None:
        return presets
    return [preset for preset in presets if preset.experiment == experiment]


def format_preset(preset: MnistSumParityPreset) -> str:
    return (
        f"{preset.name} | experiment={preset.experiment} | variant={preset.config_variant} | "
        f"mode={preset.config_mode} | n_digits={preset.n_digits} | "
        f"T={preset.reasoning_steps} | epochs={preset.epochs} | batch_size={preset.batch_size} | "
        f"ilp_chunk_size={preset.ilp_chunk_size} | lambda_mode={preset.lambda_mode} | "
        f"lam=({preset.lam0}, {preset.lam1}, {preset.lam2}) | notes={preset.notes}"
    )
