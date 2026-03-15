from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal


HalfMnistExperiment = Literal["addition", "peano"]
HalfMnistMode = Literal["tight", "medium"]
LambdaMode = Literal["fixed", "schedule"]


@dataclass(frozen=True)
class HalfMnistPreset:
    name: str
    experiment: HalfMnistExperiment
    config_variant: str
    config_mode: HalfMnistMode
    reasoning_steps: int
    epochs: int
    batch_size: int
    ilp_chunk_size: int
    lambda_mode: LambdaMode
    lam0: float
    lam1: float
    lam2: float
    notes: str = ""


HALF_MNIST_PRESETS: Dict[str, HalfMnistPreset] = {
    # Addition presets ---------------------------------------------------------
    "add_medium_v1": HalfMnistPreset(
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
        notes="Original differentiable addition setup on HalfMNIST with the wider medium bias.",
    ),
    "add_tight_v1": HalfMnistPreset(
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
        notes="Same addition experiment with the tighter symbolic search space.",
    ),
    "add_canonical_v1": HalfMnistPreset(
        name="add_canonical_v1",
        experiment="addition",
        config_variant="canonical_only",
        config_mode="tight",
        reasoning_steps=2,
        epochs=30,
        batch_size=64,
        ilp_chunk_size=16,
        lambda_mode="fixed",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Addition experiment restricted to the canonical intended clause wiring only.",
    ),
    "add_broad_search_v1": HalfMnistPreset(
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
        notes="Addition experiment with a broader symbolic search space and weaker clause filtering.",
    ),
    "add_extra_tmp2_v1": HalfMnistPreset(
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
        notes="Addition experiment with an extra auxiliary predicate tmp2 available to the learner.",
    ),

    # Peano presets ------------------------------------------------------------
    "peano_medium_v1": HalfMnistPreset(
        name="peano_medium_v1",
        experiment="peano",
        config_variant="base",
        config_mode="medium",
        reasoning_steps=8,
        epochs=30,
        batch_size=64,
        ilp_chunk_size=4,
        lambda_mode="schedule",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Original Peano setup. Strongest search space, but also the heaviest.",
    ),
    "peano_tight_v1": HalfMnistPreset(
        name="peano_tight_v1",
        experiment="peano",
        config_variant="base",
        config_mode="tight",
        reasoning_steps=8,
        epochs=30,
        batch_size=64,
        ilp_chunk_size=4,
        lambda_mode="schedule",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Peano setup with tighter clause filters and unchanged reasoning depth.",
    ),
    "peano_tight_memsafe_v1": HalfMnistPreset(
        name="peano_tight_memsafe_v1",
        experiment="peano",
        config_variant="base",
        config_mode="tight",
        reasoning_steps=4,
        epochs=30,
        batch_size=16,
        ilp_chunk_size=2,
        lambda_mode="schedule",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Safer preset for fitting the Peano run on smaller GPUs without going all the way to chunk size 1.",
    ),
    "peano_medium_oomsafe_v1": HalfMnistPreset(
        name="peano_medium_oomsafe_v1",
        experiment="peano",
        config_variant="base",
        config_mode="medium",
        reasoning_steps=8,
        epochs=30,
        batch_size=8,
        ilp_chunk_size=1,
        lambda_mode="schedule",
        lam0=1.0,
        lam1=0.2,
        lam2=0.0,
        notes="Fallback preset when medium Peano otherwise runs out of memory. Slow but conservative.",
    ),
}


def get_preset(name: str) -> HalfMnistPreset:
    try:
        return HALF_MNIST_PRESETS[name]
    except KeyError as exc:
        known = ", ".join(HALF_MNIST_PRESETS.keys())
        raise KeyError(f"Unknown HalfMNIST preset '{name}'. Known presets: {known}") from exc


def list_presets(experiment: HalfMnistExperiment | None = None) -> List[HalfMnistPreset]:
    presets = list(HALF_MNIST_PRESETS.values())
    if experiment is None:
        return presets
    return [preset for preset in presets if preset.experiment == experiment]


def format_preset(preset: HalfMnistPreset) -> str:
    return (
        f"{preset.name} | experiment={preset.experiment} | variant={preset.config_variant} | "
        f"mode={preset.config_mode} | "
        f"T={preset.reasoning_steps} | epochs={preset.epochs} | batch_size={preset.batch_size} | "
        f"ilp_chunk_size={preset.ilp_chunk_size} | lambda_mode={preset.lambda_mode} | "
        f"lam=({preset.lam0}, {preset.lam1}, {preset.lam2}) | notes={preset.notes}"
    )
