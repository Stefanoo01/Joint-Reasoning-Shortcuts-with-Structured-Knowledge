from __future__ import annotations
import torch


def f_clause(a: torch.Tensor, Xc: torch.Tensor) -> torch.Tensor:
    """
    Compute F_c(a) for a single clause compiled as Xc.

    a:  shape [n], float tensor
    Xc: shape [n, w, 2], long tensor of indices into a

    Returns:
      out: shape [n], float tensor
    """
    if a.dim() != 1:
        raise ValueError(f"a must have shape [n], got {tuple(a.shape)}")
    if Xc.dim() != 3 or Xc.size(-1) != 2:
        raise ValueError(f"Xc must have shape [n,w,2], got {tuple(Xc.shape)}")
    if Xc.dtype != torch.long:
        raise ValueError("Xc must be torch.long")
    if a.size(0) != Xc.size(0):
        raise ValueError(f"n mismatch: a has {a.size(0)} but Xc has {Xc.size(0)}")

    i1 = Xc[:, :, 0]  # [n,w]
    i2 = Xc[:, :, 1]  # [n,w]

    # Advanced indexing does gather
    y1 = a[i1]  # [n,w]
    y2 = a[i2]  # [n,w]

    z = y1 * y2  # product t-norm AND

    out = z.max(dim=1).values  # existential max over assignments

    return out


def soft_or(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Smooth OR: 1 - (1-u)(1-v)
    Assumes u,v in [0,1].
    """
    if u.shape != v.shape:
        raise ValueError(f"shape mismatch: {tuple(u.shape)} vs {tuple(v.shape)}")
    return 1.0 - (1.0 - u) * (1.0 - v)


def infer_one_step(a: torch.Tensor, Xcs: list[torch.Tensor]) -> torch.Tensor:
    """
    Apply a set of clause operators in parallel for one inference step.
    a:   [n] float
    Xcs: list of [n,w,2] long

    Returns a_next: [n] float
    """
    if len(Xcs) == 0:
        return a

    Fs = [f_clause(a, Xc) for Xc in Xcs]               # list of [n]
    derived = torch.stack(Fs, dim=0).max(dim=0).values # [n]
    a_next = soft_or(a, derived)
    return a_next


def infer_T(a0: torch.Tensor, Xcs: list[torch.Tensor], T: int) -> torch.Tensor:
    """
    Iterate infer_one_step T times.
    """
    if T <= 0:
        raise ValueError("T must be > 0")
    a = a0
    for _ in range(T):
        a = infer_one_step(a, Xcs)
    return a