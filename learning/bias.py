from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Optional, Callable

from learning.data import PredicateKey
from logic.clauses import Clause


@dataclass(frozen=True)
class BiasConfig:
    """
    General ILP-style bias:
    - allowed predicates in the body for a given head predicate
    - require recursion (if desired)
    - require body connectivity (body atoms share at least one var)
    - custom per-predicate clause filters (mode/binding constraints)
    """
    allowed_body_preds: Dict[PredicateKey, Set[str]] = field(default_factory=dict)
    require_recursive: Dict[PredicateKey, bool] = field(default_factory=dict)
    require_body_connected: bool = True
    forbidden_body_preds: Dict[PredicateKey, Set[str]] = field(default_factory=dict)

    # NEW: custom filters applied after generic filters
    custom_clause_filters: Dict[PredicateKey, Callable[[Clause], bool]] = field(default_factory=dict)


def clause_body_preds(cl: Clause) -> Set[str]:
    return {b.pred for b in cl.body}


def apply_bias(
    clauses: List[Clause],
    head_key: PredicateKey,
    *,
    bias: BiasConfig,
    require_recursive_for_set: bool = False,
    recursive_pred_name: Optional[str] = None,
) -> List[Clause]:
    """
    Filter clauses for a given head predicate key.
    - If require_recursive_for_set=True, require that recursive_pred_name appears in body
      (defaults to head predicate name).
    - Apply optional custom mode/binding filter at the end.
    """
    name, arity = head_key
    allowed = bias.allowed_body_preds.get(head_key, None)
    forbidden = bias.forbidden_body_preds.get(head_key, set())
    require_rec = bias.require_recursive.get(head_key, False) or require_recursive_for_set

    rec_name = recursive_pred_name if recursive_pred_name is not None else name
    custom = bias.custom_clause_filters.get(head_key, None)

    out: List[Clause] = []
    for c in clauses:
        preds = clause_body_preds(c)

        # allowed body preds restriction
        if allowed is not None and not preds.issubset(allowed):
            continue

        # forbidden preds restriction
        if len(preds.intersection(forbidden)) > 0:
            continue

        # recursion requirement
        if require_rec and not any(b.pred == rec_name for b in c.body):
            continue

        # body connectivity requirement
        if bias.require_body_connected and not c.is_body_connected():
            continue

        # custom mode/binding filter
        if custom is not None and not custom(c):
            continue

        out.append(c)

    return out