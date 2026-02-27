# learning/build_program.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List

from learning.data import PredicateKey
from learning.bias import BiasConfig, apply_bias
from learning.compile_cache import compile_clause_set_stack
from learning.model import PredicateClauseCache

from logic.atoms import Predicate
from logic.clauses import Clause
from logic.templates import ProgramTemplate
from logic.clauses import generate_clauses_for_template


@dataclass(frozen=True)
class ClauseSets:
    C1: List[Clause]
    C2: List[Clause]


def build_clause_sets_for_program(
    *,
    predicates: List[Predicate],
    target_pred: Predicate,
    program: ProgramTemplate,
) -> Dict[PredicateKey, ClauseSets]:
    """
    Generate (unfiltered) candidate clause sets for each intensional predicate in Π.
    """
    out: Dict[PredicateKey, ClauseSets] = {}
    intensional = program.intensional_predicates(target_pred)
    for p in intensional:
        tau1, tau2 = program.rules[(p.name, p.arity)]
        C1 = generate_clauses_for_template(head_pred=p, predicates=predicates, template=tau1)
        C2 = generate_clauses_for_template(head_pred=p, predicates=predicates, template=tau2)
        out[(p.name, p.arity)] = ClauseSets(C1=C1, C2=C2)
    return out


def build_caches_with_bias(
    *,
    clause_sets: Dict[PredicateKey, ClauseSets],
    constants: List[str],
    atom_to_idx,
    n: int,
    bot_idx: int,
    bias: BiasConfig,
    # Optional: mark which (head_key) uses recursion requirement on its C2 set
    require_recursive_on_C2: Dict[PredicateKey, bool] | None = None,
) -> Tuple[Dict[PredicateKey, PredicateClauseCache], Dict[PredicateKey, Tuple[List[str], List[str]]]]:
    """
    Apply bias and compile clause sets into cached X stacks.
    Returns:
      caches[p] = (X1_stack, X2_stack)
      clause_texts[p] = (list[str] C1, list[str] C2) after filtering
    """
    caches: Dict[PredicateKey, PredicateClauseCache] = {}
    texts: Dict[PredicateKey, Tuple[List[str], List[str]]] = {}

    require_recursive_on_C2 = require_recursive_on_C2 or {}

    for head_key, sets in clause_sets.items():
        # Apply bias to C1 always (no recursion requirement forced)
        C1_f = apply_bias(sets.C1, head_key, bias=bias, require_recursive_for_set=False)

        # Apply bias to C2, with optional recursion requirement
        C2_f = apply_bias(
            sets.C2,
            head_key,
            bias=bias,
            require_recursive_for_set=require_recursive_on_C2.get(head_key, False),
            recursive_pred_name=head_key[0],
        )

        if len(C1_f) == 0 or len(C2_f) == 0:
            raise ValueError(f"After bias, empty clause set for {head_key}: |C1|={len(C1_f)} |C2|={len(C2_f)}")

        X1 = compile_clause_set_stack(C1_f, constants, atom_to_idx, n=n, bot_idx=bot_idx)
        X2 = compile_clause_set_stack(C2_f, constants, atom_to_idx, n=n, bot_idx=bot_idx)

        caches[head_key] = PredicateClauseCache(X1=X1, X2=X2)
        texts[head_key] = ([str(c) for c in C1_f], [str(c) for c in C2_f])

    return caches, texts