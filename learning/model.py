from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
from learning.data import PredicateKey


@dataclass(frozen=True)
class PredicateClauseCache:
    X1: torch.Tensor  # [m1,n,w1,2]
    X2: torch.Tensor  # [m2,n,w2,2]


def f_clause_stack(a: torch.Tensor, X_stack: torch.Tensor) -> torch.Tensor:
    """
    a: [n] float
    X_stack: [m,n,w,2] long
    returns: [m,n] float where out[j] = f_clause(a, X_stack[j])
    """
    if a.dim() != 1:
        raise ValueError("a must be [n]")
    if X_stack.dim() != 4 or X_stack.size(-1) != 2:
        raise ValueError("X_stack must be [m,n,w,2]")
    if X_stack.dtype != torch.long:
        raise ValueError("X_stack must be long")
    m, n, w, _ = X_stack.shape
    if a.size(0) != n:
        raise ValueError("n mismatch")

    i1 = X_stack[:, :, :, 0]      # [m,n,w]
    i2 = X_stack[:, :, :, 1]      # [m,n,w]

    # gather: broadcast a -> indexing gives [m,n,w]
    y1 = a[i1]                    # [m,n,w]
    y2 = a[i2]                    # [m,n,w]
    z = y1 * y2                   # [m,n,w]
    out = z.max(dim=2).values     # max over w -> [m,n]
    return out


class ProgramLearner(nn.Module):
    """
    Paper-like: for each intensional predicate p we have two templates -> two clause sets C1,C2
    and we learn W_p over pairs (j,k).
    """
    def __init__(
        self,
        caches: Dict[PredicateKey, PredicateClauseCache],
        init_scale: float = 1e-2,
    ):
        super().__init__()
        self.caches = caches

        # Trainable W per predicate key
        self.W: nn.ParameterDict = nn.ParameterDict()
        for (name, arity), cache in caches.items():
            m1 = cache.X1.shape[0]
            m2 = cache.X2.shape[0]
            # small random init -> near-uniform softmax initially
            param = nn.Parameter(init_scale * torch.randn(m1, m2))
            self.W[self._key(name, arity)] = param

    @staticmethod
    def _key(name: str, arity: int) -> str:
        return f"{name}/{arity}"

    def get_W(self, name: str, arity: int) -> torch.Tensor:
        return self.W[self._key(name, arity)]

    def predicate_forward(
        self,
        a: torch.Tensor,
        pred_name: str,
        arity: int,
        temperature: float = 1.0,
        fast: bool = True,
    ) -> torch.Tensor:
        """
        Compute F_p(a) for one intensional predicate using W over clause pairs.
        Returns [n]. If fast=True, avoids materializing [m1,m2,n].
        """
        key = (pred_name, arity)
        cache = self.caches[key]
        W = self.get_W(pred_name, arity)  # [m1,m2]

        F1 = f_clause_stack(a, cache.X1)  # [m1,n]
        F2 = f_clause_stack(a, cache.X2)  # [m2,n]

        # pi over pairs
        logits = (W / temperature).reshape(-1)
        pi = torch.softmax(logits, dim=0).reshape_as(W)  # [m1,m2]

        if not fast:
            OR_pairs = 1.0 - (1.0 - F1[:, None, :]) * (1.0 - F2[None, :, :])  # [m1,m2,n]
            Fp = (pi[:, :, None] * OR_pairs).sum(dim=(0, 1))  # [n]
            return Fp

        # ---- FAST exact computation using soft_or(u,v)=u+v-u*v ----
        # E[u] = sum_j pi1[j]*F1[j]
        pi1 = pi.sum(dim=1)  # [m1]
        pi2 = pi.sum(dim=0)  # [m2]

        Eu = (pi1[:, None] * F1).sum(dim=0)  # [n]
        Ev = (pi2[:, None] * F2).sum(dim=0)  # [n]

        # E[u*v] = sum_{j,k} pi[j,k] * F1[j]*F2[k]
        # Compute M = pi @ F2  => [m1,n]
        M = pi @ F2  # [m1,n]
        Euv = (F1 * M).sum(dim=0)  # [n]

        Fp = Eu + Ev - Euv
        return Fp

    def infer_one_step_paper(
        self,
        a: torch.Tensor,
        temperature: float = 1.0,
        fast: bool = True,
    ) -> torch.Tensor:
        derived_list = []
        for (name, arity) in self.caches.keys():
            derived_list.append(self.predicate_forward(a, name, arity, temperature=temperature, fast=fast))
        derived = torch.stack(derived_list, dim=0).max(dim=0).values  # [n]
        a_next = 1.0 - (1.0 - a) * (1.0 - derived)
        return a_next

    def infer_T_paper(
        self,
        a0: torch.Tensor,
        T: int,
        temperature: float = 1.0,
        fast: bool = True,
    ) -> torch.Tensor:
        a = a0
        for _ in range(T):
            a = self.infer_one_step_paper(a, temperature=temperature, fast=fast)
        return a


def bce_pos_neg(pred: torch.Tensor, pos_idx: list[int], neg_idx: list[int], eps: float = 1e-6) -> torch.Tensor:
    """
    pred: [n] in [0,1]
    Returns scalar BCE over pos/neg sets.
    """
    pos = pred[pos_idx].clamp(eps, 1 - eps)
    neg = pred[neg_idx].clamp(eps, 1 - eps)
    loss_pos = -(pos.log()).mean()
    loss_neg = -((1 - neg).log()).mean()
    return loss_pos + loss_neg

def pair_distribution_entropy(W: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    logits = (W / temperature).reshape(-1)
    pi = torch.softmax(logits, dim=0)
    ent = -(pi * (pi.clamp(1e-9).log())).sum()
    return ent