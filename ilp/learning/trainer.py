# learning/trainer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import torch.nn as nn

from ilp.learning.data import PredicateKey, Targets
from ilp.learning.model import ProgramLearner, bce_pos_neg, pair_distribution_entropy
from ilp.learning.examples import Example
from ilp.logic.valuation_soft import build_a0_from_facts


@dataclass
class TrainConfig:
    epochs: int = 400
    lr: float = 5e-2
    temperature_start: float = 2.0
    temperature_end: float = 0.2
    entropy_coeff: float = 1e-3
    weight_decay: float = 0.0
    log_every: int = 50


def linear_anneal(t0: float, t1: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return t1
    alpha = step / (total_steps - 1)
    return (1 - alpha) * t0 + alpha * t1


def top_pair_index(W: torch.Tensor, temperature: float = 1.0) -> Tuple[int, int, float]:
    """
    Return (j,k,prob) of the most likely clause pair under softmax(W/temp).
    """
    logits = (W / temperature).reshape(-1)
    pi = torch.softmax(logits, dim=0)
    idx = int(torch.argmax(pi).item())
    prob = float(pi[idx].item())
    m2 = W.shape[1]
    j = idx // m2
    k = idx % m2
    return j, k, prob


def top_k_pair_indices(
    W: torch.Tensor,
    k: int,
    temperature: float = 1.0,
) -> List[Tuple[int, int, float]]:
    """
    Return the top-k clause pairs (j, k, prob) under softmax(W/temp).
    """
    if k <= 0:
        return []

    logits = (W / temperature).reshape(-1)
    pi = torch.softmax(logits, dim=0)
    top_probs, top_idx = torch.topk(pi, k=min(k, pi.numel()))

    m2 = W.shape[1]
    out = []
    for idx, prob in zip(top_idx.tolist(), top_probs.tolist()):
        j = idx // m2
        k_idx = idx % m2
        out.append((j, k_idx, float(prob)))
    return out


@torch.no_grad()
def predicate_accuracy(aT: torch.Tensor, targets: Targets, threshold: float = 0.5) -> float:
    """
    Accuracy on the predicate atoms only.
    """
    pos = aT[targets.pos_idx] >= threshold
    neg = aT[targets.neg_idx] < threshold
    correct = int(pos.sum().item()) + int(neg.sum().item())
    total = len(targets.pos_idx) + len(targets.neg_idx)
    return correct / total


def train_program(
    learner: ProgramLearner,
    a0_batch: torch.Tensor,  # [B,n] or [n]
    T: int,
    target_key: PredicateKey,
    targets: Targets,
    cfg: TrainConfig,
    clause_texts: Dict[PredicateKey, Tuple[List[str], List[str]]] | None = None,
) -> None:
    """
    End-to-end training:
    - infer_T_paper over all intensional predicates (including aux)
    - loss on target predicate only (BCE pos/neg)
    - entropy regularization on all W (semi-latent, paper-like)
    """
    learner.train()
    opt = torch.optim.Adam(learner.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if a0_batch.dim() == 1:
        a0_batch = a0_batch.unsqueeze(0)  # [1,n]
    if a0_batch.dim() != 2:
        raise ValueError("a0_batch must be [n] or [B,n]")

    Bsz, n = a0_batch.shape
    total_steps = cfg.epochs

    for step in range(total_steps):
        temp = linear_anneal(cfg.temperature_start, cfg.temperature_end, step, total_steps)
        opt.zero_grad()

        # forward for each example, average loss
        loss_main = 0.0
        for b in range(Bsz):
            aT = learner.infer_T_paper(a0_batch[b], T=T, temperature=temp, fast=True)
            loss_main = loss_main + bce_pos_neg(aT, targets.pos_idx, targets.neg_idx)
        loss_main = loss_main / Bsz

        # entropy reg on all predicates
        ent = 0.0
        for (name, arity) in learner.caches.keys():
            W = learner.get_W(name, arity)
            ent = ent + pair_distribution_entropy(W, temperature=temp)

        loss = loss_main + cfg.entropy_coeff * ent
        loss.backward()
        opt.step()

        if (step % cfg.log_every) == 0 or step == total_steps - 1:
            learner.eval()
            with torch.no_grad():
                # eval on first element + average acc
                accs = []
                for b in range(Bsz):
                    aT_eval = learner.infer_T_paper(a0_batch[b], T=T, temperature=temp, fast=True)
                    accs.append(predicate_accuracy(aT_eval, targets, threshold=0.5))
                acc = sum(accs) / len(accs)
                print(f"[{step:4d}/{total_steps}] temp={temp:.3f} loss_even={float(loss_main.item()):.4f} acc={acc:.3f}")

                for key in learner.caches.keys():
                    W = learner.get_W(*key)
                    j, k, prob = top_pair_index(W, temperature=temp)
                    name, arity = key
                    msg = f"  top pair {name}/{arity}: ({j},{k}) prob={prob:.3f}"
                    if clause_texts is not None and key in clause_texts:
                        C1_txt, C2_txt = clause_texts[key]
                        msg += f"\n    C1: {C1_txt[j]}\n    C2: {C2_txt[k]}"
                    print(msg)
            learner.train()

@torch.no_grad()
def extract_hard_program(
    learner: ProgramLearner,
    temperature: float = 0.2,
) -> Dict[PredicateKey, Tuple[int, int, float]]:
    """
    Return the best (j,k,prob) per predicate under current W.
    """
    out = {}
    for key in learner.caches.keys():
        W = learner.get_W(*key)
        j, k, prob = top_pair_index(W, temperature=temperature)
        out[key] = (j, k, prob)
    return out


@torch.no_grad()
def extract_topk_program(
    learner: ProgramLearner,
    k: int = 5,
    temperature: float = 0.2,
) -> Dict[PredicateKey, List[Tuple[int, int, float]]]:
    """
    Return the top-k clause pairs per predicate under current W.
    """
    out = {}
    for key in learner.caches.keys():
        W = learner.get_W(*key)
        out[key] = top_k_pair_indices(W, k=k, temperature=temperature)
    return out

from typing import Dict
from ilp.logic.atoms import Atom
from ilp.learning.data import PredicateKey
from ilp.learning.examples import Example
from ilp.logic.valuation_soft import build_a0_from_facts


def train_program_examples(
    *,
    learner: ProgramLearner,
    examples: List[Example],
    atom_to_idx: Dict[Atom, int],
    n: int,
    bot_idx: int,
    T: int,
    cfg: TrainConfig,
    clause_texts: Dict[PredicateKey, Tuple[List[str], List[str]]] | None = None,
    device: torch.device | None = None,
) -> None:
    """
    Production-style trainer:
      - each Example carries its own targets (pos/neg) and its own hard/soft facts
      - loss is averaged across examples
      - aux predicates remain latent (no direct targets unless you include them in Example.targets)
    """
    if device is None:
        device = torch.device("cpu")

    learner.to(device)
    learner.train()
    opt = torch.optim.Adam(learner.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs
    Bsz = len(examples)
    if Bsz == 0:
        raise ValueError("examples must be non-empty")

    for step in range(total_steps):
        temp = linear_anneal(cfg.temperature_start, cfg.temperature_end, step, total_steps)

        opt.zero_grad()

        # ---- main loss: average over examples ----
        loss_main = 0.0
        for ex in examples:
            a0 = build_a0_from_facts(
                n=n,
                atom_to_idx=atom_to_idx,
                bot_idx=bot_idx,
                hard_facts=ex.hard_facts,
                soft_facts=ex.soft_facts,
                device=device,
            )
            aT = learner.infer_T_paper(a0, T=T, temperature=temp, fast=True)
            loss_main = loss_main + bce_pos_neg(aT, ex.targets.pos_idx, ex.targets.neg_idx)
        loss_main = loss_main / Bsz

        # ---- entropy reg on all predicates ----
        ent = 0.0
        for (name, arity) in learner.caches.keys():
            W = learner.get_W(name, arity)
            ent = ent + pair_distribution_entropy(W, temperature=temp)

        loss = loss_main + cfg.entropy_coeff * ent
        loss.backward()
        opt.step()

        # ---- logging ----
        if (step % cfg.log_every) == 0 or step == total_steps - 1:
            learner.eval()
            with torch.no_grad():
                # accuracy averaged over examples (on their own targets)
                accs = []
                for ex in examples:
                    a0 = build_a0_from_facts(
                        n=n,
                        atom_to_idx=atom_to_idx,
                        bot_idx=bot_idx,
                        hard_facts=ex.hard_facts,
                        soft_facts=ex.soft_facts,
                        device=device,
                    )
                    aT_eval = learner.infer_T_paper(a0, T=T, temperature=temp, fast=True)
                    accs.append(predicate_accuracy(aT_eval, ex.targets, threshold=0.5))
                acc = sum(accs) / len(accs)

                print(f"[{step:4d}/{total_steps}] temp={temp:.3f} loss={float(loss_main.item()):.4f} acc={acc:.3f}")

                for key in learner.caches.keys():
                    W = learner.get_W(*key)
                    j, k, prob = top_pair_index(W, temperature=temp)
                    name, arity = key
                    msg = f"  top pair {name}/{arity}: ({j},{k}) prob={prob:.3f}"
                    if clause_texts is not None and key in clause_texts:
                        C1_txt, C2_txt = clause_texts[key]
                        msg += f"\n    C1: {C1_txt[j]}\n    C2: {C2_txt[k]}"
                    print(msg)

            learner.train()
