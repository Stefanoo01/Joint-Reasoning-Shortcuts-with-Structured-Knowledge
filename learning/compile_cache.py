from __future__ import annotations
from typing import Dict, List
import torch
from logic.atoms import Atom
from logic.clauses import Clause
from compile.compile_clause import compile_clause_to_X


def compile_clause_set_stack(
    clauses: List[Clause],
    constants: List[str],
    atom_to_idx: Dict[Atom, int],
    n: int,
    bot_idx: int,
) -> torch.Tensor:
    """
    Compile a set of clauses into a stacked tensor [m,n,w_max,2] (long).
    If clauses have different w, we pad to w_max using BOT indices (bot_idx, bot_idx).
    This is safe because a[bot_idx]=0, so padding doesn't affect product/max.
    """
    compiled: List[torch.Tensor] = []
    w_max = 0

    # 1) compile each clause
    for c in clauses:
        X_py = compile_clause_to_X(
            clause=c,
            constants=constants,
            atom_to_idx=atom_to_idx,
            n=n,
            bot_idx=bot_idx,
        )
        X_t = torch.tensor(X_py, dtype=torch.long)  # [n,w,2]
        compiled.append(X_t)
        w_max = max(w_max, X_t.shape[1])

    # 2) pad to w_max
    padded: List[torch.Tensor] = []
    for X in compiled:
        w = X.shape[1]
        if w < w_max:
            pad = torch.full((n, w_max - w, 2), bot_idx, dtype=torch.long)
            X = torch.cat([X, pad], dim=1)  # [n,w_max,2]
        padded.append(X)

    return torch.stack(padded, dim=0)  # [m,n,w_max,2]