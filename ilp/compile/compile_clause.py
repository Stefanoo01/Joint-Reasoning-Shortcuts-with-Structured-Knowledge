from __future__ import annotations
from itertools import product
from typing import Dict, List
from ilp.logic.atoms import Atom
from ilp.logic.clauses import AtomSchema, Clause, Var

Subst = Dict[Var, str]  # variable -> constant


def clause_existential_vars(clause: Clause) -> List[Var]:
    """
    Existential vars = vars that appear in the body but not in the head.
    Returned in deterministic order by variable name.
    """
    hv = clause.head_vars()
    bv = clause.body_vars()
    ex = [v for v in bv if v not in hv]
    # deterministic order
    ex.sort(key=lambda v: v.name)
    return ex


def enumerate_existential_assignments(ex_vars: List[Var], constants: List[str]) -> List[Subst]:
    """
    Enumerate all assignments for existential variables over constants.
    Deterministic order: ex_vars order then constants_sorted order.
    Returns list of dictionaries {Var: constant}.
    """
    constants_sorted = sorted(constants)

    if len(ex_vars) == 0:
        return [{}]  # w = 1

    out: List[Subst] = []
    for values in product(constants_sorted, repeat=len(ex_vars)):
        s: Subst = {}
        for v, c in zip(ex_vars, values):
            s[v] = c
        out.append(s)
    return out


def ground_atom_schema(atom_s: AtomSchema, subst: Subst) -> Atom:
    """
    Ground an AtomSchema using a substitution subst (Var -> constant string).
    """
    if atom_s.arity == 0:
        return Atom(atom_s.pred, ())
    grounded_args = tuple(subst[v] for v in atom_s.args)
    return Atom(atom_s.pred, grounded_args)

def enumerate_ground_heads_for_predicate(
    head_pred: str,
    head_arity: int,
    constants: List[str],
) -> List[Atom]:
    constants_sorted = sorted(constants)
    if head_arity == 0:
        return [Atom(head_pred, ())]
    if head_arity == 1:
        return [Atom(head_pred, (c,)) for c in constants_sorted]
    if head_arity == 2:
        out: List[Atom] = []
        for c1 in constants_sorted:
            for c2 in constants_sorted:
                out.append(Atom(head_pred, (c1, c2)))
        return out
    raise ValueError("head_arity must be 0,1,2")


def compile_clause_to_X(
    clause: Clause,
    constants: List[str],
    atom_to_idx: Dict[Atom, int],
    n: int,
    bot_idx: int,
) -> List[List[tuple[int, int]]]:
    """
    Compile a Clause into index tensor X_c with logical shape [n][w][2],
    returned as nested Python lists for now.

    X[k][t] = (i1, i2) are indices in G for the two body atoms that support head atom k
    under the t-th existential assignment.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if bot_idx < 0 or bot_idx >= n:
        raise ValueError("bot_idx out of range")

    ex_vars = clause_existential_vars(clause)
    assignments = enumerate_existential_assignments(ex_vars, constants)
    w = len(assignments)

    # initialize with BOT padding
    X: List[List[tuple[int, int]]] = [[(bot_idx, bot_idx) for _ in range(w)] for _ in range(n)]

    # enumerate all possible ground heads for this predicate
    head_atoms = enumerate_ground_heads_for_predicate(
        head_pred=clause.head.pred,
        head_arity=clause.head.arity,
        constants=constants,
    )

    # map head schema vars to ground head arguments
    head_vars = list(clause.head.args)  # tuple[Var,...] -> list
    for ha in head_atoms:
        # Some heads might not be in G (shouldn't happen if G built from same constants)
        if ha not in atom_to_idx:
            continue
        k = atom_to_idx[ha]

        # build substitution for head variables
        head_subst: Subst = {}
        for v, c in zip(head_vars, ha.args):
            head_subst[v] = c

        # fill all existential assignments
        for t, ex_subst in enumerate(assignments):
            subst_total: Subst = dict(head_subst)
            subst_total.update(ex_subst)

            b1_g = ground_atom_schema(clause.body[0], subst_total)
            b2_g = ground_atom_schema(clause.body[1], subst_total)

            i1 = atom_to_idx.get(b1_g, bot_idx)
            i2 = atom_to_idx.get(b2_g, bot_idx)

            X[k][t] = (i1, i2)

    return X