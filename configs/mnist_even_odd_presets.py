from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal


MnistEvenOddExperiment = Literal["addition"]
MnistEvenOddMode = Literal["tight", "medium"]
LambdaMode = Literal["fixed", "schedule"]


@dataclass(frozen=True)
class MnistEvenOddPreset:
    name: str
    experiment: MnistEvenOddExperiment
    config_variant: str
    config_mode: MnistEvenOddMode
    reasoning_steps: int
    epochs: int
    batch_size: int
    ilp_chunk_size: int
    lambda_mode: LambdaMode
    lam0: float
    lam1: float
    lam2: float
    notes: str = ""


MNIST_EVEN_ODD_PRESETS: Dict[str, MnistEvenOddPreset] = {
    "add_medium_v1": MnistEvenOddPreset(
        name="add_medium_v1",
        experiment="addition",
        config_variant="base",
        config_mode="medium",
        reasoning_steps=2,
        epochs=30,
        batch_size=64,
        ilp_chunk_size=16,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Base MNIST-Even-Odd addition setup with the wider medium clause bias.",
    ),
    "add_tight_v1": MnistEvenOddPreset(
        name="add_tight_v1",
        experiment="addition",
        config_variant="base",
        config_mode="tight",
        reasoning_steps=2,
        epochs=30,
        batch_size=64,
        ilp_chunk_size=16,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Same MNIST-Even-Odd addition setup with the tighter symbolic search space.",
    ),
    "add_broad_search_v1": MnistEvenOddPreset(
        name="add_broad_search_v1",
        experiment="addition",
        config_variant="broad_search",
        config_mode="medium",
        reasoning_steps=2,
        epochs=30,
        batch_size=64,
        ilp_chunk_size=16,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="MNIST-Even-Odd addition with a broader symbolic search space while keeping clause shapes structured.",
    ),
    "add_sum_relaxed_v1": MnistEvenOddPreset(
        name="add_sum_relaxed_v1",
        experiment="addition",
        config_variant="sum_relaxed",
        config_mode="medium",
        reasoning_steps=2,
        epochs=30,
        batch_size=64,
        ilp_chunk_size=16,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="MNIST-Even-Odd addition close to the base setup, but with a wider search space for sum_is.",
    ),
    "add_extra_tmp2_v1": MnistEvenOddPreset(
        name="add_extra_tmp2_v1",
        experiment="addition",
        config_variant="extra_tmp2",
        config_mode="medium",
        reasoning_steps=2,
        epochs=30,
        batch_size=64,
        ilp_chunk_size=16,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="MNIST-Even-Odd addition with an extra auxiliary predicate tmp2 available to the learner.",
    ),
}


def get_preset(name: str) -> MnistEvenOddPreset:
    try:
        return MNIST_EVEN_ODD_PRESETS[name]
    except KeyError as exc:
        known = ", ".join(MNIST_EVEN_ODD_PRESETS.keys())
        raise KeyError(
            f"Unknown MNIST-Even-Odd preset '{name}'. Known presets: {known}"
        ) from exc


def list_presets(
    experiment: MnistEvenOddExperiment | None = None,
) -> List[MnistEvenOddPreset]:
    presets = list(MNIST_EVEN_ODD_PRESETS.values())
    if experiment is None:
        return presets
    return [preset for preset in presets if preset.experiment == experiment]


def format_preset(preset: MnistEvenOddPreset) -> str:
    return (
        f"{preset.name} | experiment={preset.experiment} | variant={preset.config_variant} | "
        f"mode={preset.config_mode} | "
        f"T={preset.reasoning_steps} | epochs={preset.epochs} | batch_size={preset.batch_size} | "
        f"ilp_chunk_size={preset.ilp_chunk_size} | lambda_mode={preset.lambda_mode} | "
        f"lam=({preset.lam0}, {preset.lam1}, {preset.lam2}) | notes={preset.notes}"
    )
