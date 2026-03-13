from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
from ilp.learning.data import PredicateKey


@dataclass(frozen=True)
class PredicateClauseCache:
    X1: torch.Tensor  # [m1,n,w1,2]
    X2: torch.Tensor  # [m2,n,w2,2]


def f_clause_stack(a: torch.Tensor, X_stack: torch.Tensor) -> torch.Tensor:
    """
    a: [n] or [B,n] float
    X_stack: [m,n,w,2] long
    returns:
      - [m,n] for a=[n]
      - [B,m,n] for a=[B,n]
    """
    if X_stack.dim() != 4 or X_stack.size(-1) != 2:
        raise ValueError("X_stack must be [m,n,w,2]")
    if X_stack.dtype != torch.long:
        raise ValueError("X_stack must be long")
    m, n, w, _ = X_stack.shape

    i1 = X_stack[:, :, :, 0]      # [m,n,w]
    i2 = X_stack[:, :, :, 1]      # [m,n,w]

    if a.dim() == 1:
        if a.size(0) != n:
            raise ValueError("n mismatch")

        # gather: broadcast a -> indexing gives [m,n,w]
        y1 = a[i1]                    # [m,n,w]
        y2 = a[i2]                    # [m,n,w]
        z = y1 * y2                   # [m,n,w]
        out = z.max(dim=2).values     # max over w -> [m,n]
        return out

    if a.dim() != 2:
        raise ValueError("a must be [n] or [B,n]")
    if a.size(1) != n:
        raise ValueError("n mismatch")

    y1 = a[:, i1]                 # [B,m,n,w]
    y2 = a[:, i2]                 # [B,m,n,w]
    z = y1 * y2                   # [B,m,n,w]
    out = z.max(dim=3).values     # max over w -> [B,m,n]
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

    def _apply(self, fn):
        # Keep compiled clause caches aligned with the module device/dtype moves.
        super()._apply(fn)
        self.caches = {
            key: PredicateClauseCache(
                X1=fn(cache.X1),
                X2=fn(cache.X2),
            )
            for key, cache in self.caches.items()
        }
        return self

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
        Returns [n] or [B,n]. If fast=True, avoids materializing pair tensors.
        """
        key = (pred_name, arity)
        cache = self.caches[key]
        W = self.get_W(pred_name, arity)  # [m1,m2]

        F1 = f_clause_stack(a, cache.X1)
        F2 = f_clause_stack(a, cache.X2)

        # pi over pairs
        logits = (W / temperature).reshape(-1)
        pi = torch.softmax(logits, dim=0).reshape_as(W)  # [m1,m2]

        if not fast:
            if a.dim() == 1:
                OR_pairs = 1.0 - (1.0 - F1[:, None, :]) * (1.0 - F2[None, :, :])  # [m1,m2,n]
                Fp = (pi[:, :, None] * OR_pairs).sum(dim=(0, 1))  # [n]
                return Fp

            OR_pairs = 1.0 - (1.0 - F1[:, :, None, :]) * (1.0 - F2[:, None, :, :])  # [B,m1,m2,n]
            Fp = (pi[None, :, :, None] * OR_pairs).sum(dim=(1, 2))  # [B,n]
            return Fp

        # ---- FAST exact computation using soft_or(u,v)=u+v-u*v ----
        pi1 = pi.sum(dim=1)  # [m1]
        pi2 = pi.sum(dim=0)  # [m2]

        if a.dim() == 1:
            Eu = (pi1[:, None] * F1).sum(dim=0)  # [n]
            Ev = (pi2[:, None] * F2).sum(dim=0)  # [n]
            M = pi @ F2  # [m1,n]
            Euv = (F1 * M).sum(dim=0)  # [n]
            return Eu + Ev - Euv

        Eu = (pi1[None, :, None] * F1).sum(dim=1)  # [B,n]
        Ev = (pi2[None, :, None] * F2).sum(dim=1)  # [B,n]
        M = torch.einsum("jk,bkn->bjn", pi, F2)  # [B,m1,n]
        Euv = (F1 * M).sum(dim=1)  # [B,n]
        return Eu + Ev - Euv

    def infer_one_step_paper(
        self,
        a: torch.Tensor,
        temperature: float = 1.0,
        fast: bool = True,
    ) -> torch.Tensor:
        derived_list = []
        for (name, arity) in self.caches.keys():
            derived_list.append(self.predicate_forward(a, name, arity, temperature=temperature, fast=fast))
        stack_dim = 0 if a.dim() == 1 else 1
        derived = torch.stack(derived_list, dim=stack_dim).max(dim=stack_dim).values  # [n] or [B,n]
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
    Handles empty pos/neg sets gracefully (returns 0 for that component).
    """
    loss_pos = torch.tensor(0.0, device=pred.device)
    loss_neg = torch.tensor(0.0, device=pred.device)
    if len(pos_idx) > 0:
        pos = pred[pos_idx].clamp(eps, 1 - eps)
        loss_pos = -(pos.log()).mean()
    if len(neg_idx) > 0:
        neg = pred[neg_idx].clamp(eps, 1 - eps)
        loss_neg = -((1 - neg).log()).mean()
    return loss_pos + loss_neg

def pair_distribution_entropy(W: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    logits = (W / temperature).reshape(-1)
    pi = torch.softmax(logits, dim=0)
    ent = -(pi * (pi.clamp(1e-9).log())).sum()
    return ent
