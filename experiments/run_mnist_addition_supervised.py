from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import statistics

from logic.atoms import Atom
from logic.valuation_soft import build_a0_from_indexed_facts
from configs.mnist_addition import make_config
from learning.system_builder import build_system_from_config
from learning.model import bce_pos_neg
from learning.data import build_targets_from_positives_domains

from models.cbm_mnist import MNISTDigitCNN

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


@dataclass
class PairSample:
    img1: torch.Tensor
    img2: torch.Tensor
    d1: int
    d2: int
    s: int


def build_add_truth_table_hard_idx(atom_to_idx) -> torch.Tensor:
    """
    hard facts for add(A,B,S) for all A,B in 0..9.
    """
    hard_atoms = []
    for a in range(10):
        for b in range(10):
            s = a + b
            hard_atoms.append(Atom("add", (str(a), str(b), str(s))))
    hard_idx = torch.tensor([atom_to_idx[a] for a in hard_atoms], dtype=torch.long)
    return hard_idx


def build_digit12_soft_idx(atom_to_idx) -> torch.Tensor:
    atoms = []
    for d in range(10):
        atoms.append(Atom("digit1", (str(d),)))
    for d in range(10):
        atoms.append(Atom("digit2", (str(d),)))
    return torch.tensor([atom_to_idx[a] for a in atoms], dtype=torch.long)


def build_sum_targets(atom_to_idx, cfg, true_sum: int):
    domains = cfg.arg_domains[("sum_is", 1)]  # [[0..18]]
    return build_targets_from_positives_domains(
        atom_to_idx=atom_to_idx,
        pred_name="sum_is",
        domains=domains,
        positive_atoms=[Atom("sum_is", (str(true_sum),))],
    )

def build_sum_is_idx(atom_to_idx) -> torch.Tensor:
    atoms = [Atom("sum_is", (str(s),)) for s in range(19)]
    return torch.tensor([atom_to_idx[a] for a in atoms], dtype=torch.long)

def make_pairs(num_pairs: int, n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randrange(n), rng.randrange(n)) for _ in range(num_pairs)]

def apply_prob_noise(probs: torch.Tensor, eps: float) -> torch.Tensor:
    """
    probs: [B,10] or [10] (assume last dim is classes), values in [0,1], sum=1
    returns: noisy probs with same shape, still sums to 1
    probs' = (1-eps)*probs + eps*Uniform
    """
    if eps <= 0.0:
        return probs
    if eps >= 1.0:
        # fully uniform
        K = probs.shape[-1]
        return torch.full_like(probs, 1.0 / K)

    K = probs.shape[-1]
    uni = torch.full_like(probs, 1.0 / K)
    return (1.0 - eps) * probs + eps * uni

def apply_occlusion(
    imgs: torch.Tensor,
    frac: float,
    rng: random.Random,
) -> torch.Tensor:
    """
    imgs: [B,1,28,28] float in [0,1]
    frac: fraction of side to occlude (0..1)
    rng: python Random for reproducible placement

    Returns a NEW tensor with a black square occlusion applied (no inplace on input).
    """
    if frac <= 0.0:
        return imgs
    frac = min(frac, 1.0)

    B, C, H, W = imgs.shape
    side = int(round(frac * min(H, W)))
    side = max(1, side)

    # clone to avoid modifying original tensor
    out = imgs.clone()

    # random top-left per image
    for i in range(B):
        top = rng.randrange(0, H - side + 1) if H - side + 1 > 0 else 0
        left = rng.randrange(0, W - side + 1) if W - side + 1 > 0 else 0
        out[i, :, top:top + side, left:left + side] = 0.0

    return out

@torch.no_grad()
def evaluate(
    *,
    cbm: nn.Module,
    learner,
    dataset,
    pairs: list[tuple[int, int]],
    batch_size: int,
    device: torch.device,
    n_atoms: int,
    bot_idx: int,
    T: int,
    soft_idx_digit: torch.Tensor,   # [20]
    hard_idx_add: torch.Tensor,     # [1900]
    sum_is_idx: torch.Tensor,       # [19]
    sum_targets_cache,
    noise_eval: float = 0.0,
    occlude_eval: str = "none",
    occlude_frac: float = 0.0,
    occlude_seed: int = 123,
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

    # iterate in mini-batches
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        B = len(chunk)

        imgs1 = []
        imgs2 = []
        d1_true = []
        d2_true = []
        s_true = []

        for i1, i2 in chunk:
            img1, d1 = dataset[i1]
            img2, d2 = dataset[i2]
            imgs1.append(img1)
            imgs2.append(img2)
            d1_true.append(int(d1))
            d2_true.append(int(d2))
            s_true.append(int(d1) + int(d2))

        imgs1 = torch.stack(imgs1, dim=0).to(device)
        imgs2 = torch.stack(imgs2, dim=0).to(device)
        # --- Occlusion ONLY in evaluation ---
        occ_rng = random.Random(occlude_seed + start)  # start dipende dal batch -> placement riproducibile per batch
        if occlude_eval == "d1":
            imgs1 = apply_occlusion(imgs1, occlude_frac, occ_rng)
        elif occlude_eval == "d2":
            imgs2 = apply_occlusion(imgs2, occlude_frac, occ_rng)
        d1_true_t = torch.tensor(d1_true, dtype=torch.long, device=device)
        d2_true_t = torch.tensor(d2_true, dtype=torch.long, device=device)

        logits1 = cbm(imgs1)
        logits2 = cbm(imgs2)
        probs1 = torch.softmax(logits1, dim=1)
        probs2 = torch.softmax(logits2, dim=1)
        probs1 = apply_prob_noise(probs1, noise_eval)
        probs2 = apply_prob_noise(probs2, noise_eval)

        # concept accuracy
        d1_hat = torch.argmax(logits1, dim=1)
        d2_hat = torch.argmax(logits2, dim=1)
        correct_d1 += int((d1_hat == d1_true_t).sum().item())
        correct_d2 += int((d2_hat == d2_true_t).sum().item())

        # cbm baseline sum
        sum_hat_cbm = (d1_hat + d2_hat).detach().cpu().tolist()

        # concept loss (supervised metric)
        loss_concepts = (F.cross_entropy(logits1, d1_true_t) + F.cross_entropy(logits2, d2_true_t)) / 2.0
        loss_concepts_sum += float(loss_concepts.item()) * B

        # task loss + ilp acc (loop per sample: va bene per val; se vuoi velocizzare poi batchizziamo)
        for i in range(B):
            soft_val = torch.cat([probs1[i], probs2[i]], dim=0)  # [20]
            a0 = build_a0_from_indexed_facts(
                n=n_atoms,
                bot_idx=bot_idx,
                soft_idx=soft_idx_digit,
                soft_val=soft_val,
                hard_idx=hard_idx_add,
            )
            aT = learner.infer_T_paper(a0, T=T, temperature=1.0, fast=True)

            s = s_true[i]
            if sum_hat_cbm[i] == s:
                correct_sum_cbm += 1

            scores = aT[sum_is_idx]
            s_hat_ilp = int(torch.argmax(scores).item())
            if s_hat_ilp == s:
                correct_sum_ilp += 1

            targets = sum_targets_cache[s]
            loss_task_sum += float(bce_pos_neg(aT, targets.pos_idx, targets.neg_idx).item())

        total += B

    return {
        "loss_task": loss_task_sum / total,
        "loss_concepts": loss_concepts_sum / total,
        "acc_concepts": ((correct_d1 / total) + (correct_d2 / total)) / 2.0,
        "acc_sum_cbm": correct_sum_cbm / total,
        "acc_sum_ilp": correct_sum_ilp / total,
    }

def run_one(
    *,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lambda_mode: str,   # "fixed0" or "schedule"
    lam0: float,
    lam1: float,
    lam2: float,
    noise_eval: float = 0.0,
    occlude_eval: str = "none",
    occlude_frac: float = 0.0,
    occlude_seed: int = 123,
) -> dict:
    # -------------------------
    # 0) Reproducibility
    # -------------------------
    random.seed(seed)
    torch.manual_seed(seed)
    # (opzionale) se usi CUDA:
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    device = torch.device("cpu")

    # -------------------------
    # 1) Build ILP system
    # -------------------------
    cfg = make_config()
    bundle = build_system_from_config(cfg)
    learner = bundle.learner.to(device)

    n_atoms = len(bundle.G)
    bot_idx = bundle.bot_idx
    atom_to_idx = bundle.atom_to_idx

    # Precompute indices / caches
    hard_idx_add = build_add_truth_table_hard_idx(atom_to_idx).to(device)
    soft_idx_digit = build_digit12_soft_idx(atom_to_idx).to(device)  # digit1(0..9)+digit2(0..9)
    sum_is_idx = build_sum_is_idx(atom_to_idx).to(device)

    # Precompute targets cache for sums 0..18
    sum_targets_cache = [build_sum_targets(atom_to_idx, cfg, true_sum=s) for s in range(19)]

    # -------------------------
    # 2) CBM model + optimizer
    # -------------------------
    cbm = MNISTDigitCNN().to(device)
    opt = torch.optim.Adam(list(cbm.parameters()) + list(learner.parameters()), lr=1e-3)

    # -------------------------
    # 3) Load MNIST train/test
    # -------------------------

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = MNIST(root="data_root", train=True, download=True, transform=tfm)
    val_ds = MNIST(root="data_root", train=False, download=True, transform=tfm)

    # Val pairs: keep FIXED across seeds for fair comparison
    val_pairs = make_pairs(num_pairs=2000, n=len(val_ds), seed=123)

    # Training sampler RNG: varies with seed
    rng = random.Random(seed)

    def sample_batch():
        # returns (imgs1, imgs2, d1_true, d2_true, s_true_list)
        imgs1, imgs2 = [], []
        d1_true, d2_true = [], []
        s_true = []
        for _ in range(batch_size):
            i1 = rng.randrange(len(train_ds))
            i2 = rng.randrange(len(train_ds))
            img1, d1 = train_ds[i1]
            img2, d2 = train_ds[i2]
            imgs1.append(img1)
            imgs2.append(img2)
            d1_true.append(int(d1))
            d2_true.append(int(d2))
            s_true.append(int(d1) + int(d2))
        imgs1 = torch.stack(imgs1, dim=0).to(device)
        imgs2 = torch.stack(imgs2, dim=0).to(device)
        d1_true_t = torch.tensor(d1_true, dtype=torch.long, device=device)
        d2_true_t = torch.tensor(d2_true, dtype=torch.long, device=device)
        return imgs1, imgs2, d1_true_t, d2_true_t, s_true

    # Lambda schedule helper
    def lambda_for_epoch(ep: int) -> float:
        if lambda_mode == "fixed0":
            return 0.0
        # "schedule"
        if ep == 0:
            return lam0
        if ep == 1:
            return lam1
        if ep == 2:
            return lam2
        return 0.0

    # -------------------------
    # 4) Train loop
    # -------------------------
    for ep in range(epochs):
        lam = lambda_for_epoch(ep)
        cbm.train()
        learner.train()

        for step in range(steps_per_epoch):
            imgs1, imgs2, d1_true_t, d2_true_t, s_true = sample_batch()

            logits1 = cbm(imgs1)  # [B,10]
            logits2 = cbm(imgs2)  # [B,10]
            probs1 = torch.softmax(logits1, dim=1)
            probs2 = torch.softmax(logits2, dim=1)

            # concept loss (we keep it computed even if lam=0; it’s cheap and useful)
            loss_concepts = (F.cross_entropy(logits1, d1_true_t) + F.cross_entropy(logits2, d2_true_t)) / 2.0

            # task loss via ILP
            loss_task = 0.0
            for i in range(batch_size):
                soft_val = torch.cat([probs1[i], probs2[i]], dim=0)  # [20]

                a0 = build_a0_from_indexed_facts(
                    n=n_atoms,
                    bot_idx=bot_idx,
                    soft_idx=soft_idx_digit,
                    soft_val=soft_val,
                    hard_idx=hard_idx_add,
                )

                aT = learner.infer_T_paper(a0, T=bundle.program.T, temperature=1.0, fast=True)

                targets = sum_targets_cache[s_true[i]]
                loss_task = loss_task + bce_pos_neg(aT, targets.pos_idx, targets.neg_idx)

            loss_task = loss_task / batch_size
            loss = loss_task + lam * loss_concepts

            opt.zero_grad()
            loss.backward()
            opt.step()

    # -------------------------
    # 5) Final VAL evaluation
    # -------------------------
    val_metrics = evaluate(
        cbm=cbm,
        learner=learner,
        dataset=val_ds,
        pairs=val_pairs,
        batch_size=batch_size,
        device=device,
        n_atoms=n_atoms,
        bot_idx=bot_idx,
        T=bundle.program.T,
        soft_idx_digit=soft_idx_digit,
        hard_idx_add=hard_idx_add,
        sum_is_idx=sum_is_idx,
        sum_targets_cache=sum_targets_cache,
        noise_eval=noise_eval,
        occlude_eval=occlude_eval,
        occlude_frac=occlude_frac,
        occlude_seed=occlude_seed,
    )

    return {
        "val_acc_concepts": val_metrics["acc_concepts"],
        "val_acc_sum_ilp": val_metrics["acc_sum_ilp"],
        "val_acc_sum_cbm": val_metrics["acc_sum_cbm"],
        "val_loss_task": val_metrics["loss_task"],
        "val_loss_concepts": val_metrics["loss_concepts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    # Training params
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)

    # Sweep params
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=5)

    # Lambda mode
    parser.add_argument("--lambda_mode", type=str, choices=["fixed0", "schedule"], default="schedule")
    parser.add_argument("--lam0", type=float, default=1.0)
    parser.add_argument("--lam1", type=float, default=0.2)
    parser.add_argument("--lam2", type=float, default=0.0)

    parser.add_argument("--noise_eval", type=float, default=0.0,
                    help="Apply probability noise ONLY in evaluation: probs'=(1-eps)*probs+eps*U")
    
    parser.add_argument("--occlude_eval", type=str, default="none", choices=["none", "d1", "d2"],
                    help="Occlude ONLY in evaluation: none|d1|d2")
    parser.add_argument("--occlude_frac", type=float, default=0.0,
                        help="Occlusion size as fraction of image side (0..1). Example 0.3 -> ~8px on 28x28")
    parser.add_argument("--occlude_seed", type=int, default=123,
                        help="Seed for occlusion placement in evaluation")

    args = parser.parse_args()

    results = []
    print(
        f"=== SWEEP START ===\n"
        f"lambda_mode={args.lambda_mode}  seeds={args.num_seeds}  seed_start={args.seed_start}\n"
        f"epochs={args.epochs} steps_per_epoch={args.steps_per_epoch} batch_size={args.batch_size}\n"
        f"lam0={args.lam0} lam1={args.lam1} lam2={args.lam2}\n"
    )

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        out = run_one(
            seed=seed,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            lambda_mode=args.lambda_mode,
            lam0=args.lam0,
            lam1=args.lam1,
            lam2=args.lam2,
            noise_eval=args.noise_eval,
            occlude_eval=args.occlude_eval,
            occlude_frac=args.occlude_frac,
            occlude_seed=args.occlude_seed,
        )
        results.append((seed, out))
        print(
            f"[SEED {seed}] "
            f"val_acc_sum_ilp={out['val_acc_sum_ilp']:.3f} "
            f"val_acc_concepts={out['val_acc_concepts']:.3f} "
            f"val_loss_task={out['val_loss_task']:.4f}"
        )

    # Summary
    acc_sum = [o["val_acc_sum_ilp"] for _, o in results]
    acc_con = [o["val_acc_concepts"] for _, o in results]
    loss_task = [o["val_loss_task"] for _, o in results]

    def _summary(xs):
        return {
            "mean": statistics.mean(xs),
            "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs),
            "max": max(xs),
        }

    s_sum = _summary(acc_sum)
    s_con = _summary(acc_con)
    s_loss = _summary(loss_task)

    print("\n=== SUMMARY ===")
    print(f"val_acc_sum_ilp: mean={s_sum['mean']:.3f} std={s_sum['std']:.3f} min={s_sum['min']:.3f} max={s_sum['max']:.3f}")
    print(f"val_acc_concepts: mean={s_con['mean']:.3f} std={s_con['std']:.3f} min={s_con['min']:.3f} max={s_con['max']:.3f}")
    print(f"val_loss_task: mean={s_loss['mean']:.4f} std={s_loss['std']:.4f} min={s_loss['min']:.4f} max={s_loss['max']:.4f}")

    print("\n=== SWEEP END ===")


if __name__ == "__main__":
    main()