from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from logic.atoms import Atom

PredicateKey = Tuple[str, int]


@dataclass(frozen=True)
class Targets:
    all_idx: List[int]
    pos_idx: List[int]
    neg_idx: List[int]


def predicate_ground_atoms(constants: List[str], pred_name: str, arity: int) -> List[Atom]:
    C = sorted(constants)
    if arity == 0:
        return [Atom(pred_name, ())]
    if arity == 1:
        return [Atom(pred_name, (c,)) for c in C]
    if arity == 2:
        return [Atom(pred_name, (a, b)) for a in C for b in C]
    if arity == 3:
        return [Atom(pred_name, (a, b, c)) for a in C for b in C for c in C]
    raise ValueError("Only arity 0/1/2 supported in this toy implementation")


def build_targets_from_positives(
    atom_to_idx: Dict[Atom, int],
    constants: List[str],
    pred_name: str,
    arity: int,
    positive_atoms: List[Atom],
) -> Targets:
    """
    General: targets for a predicate p/arity given positives.
    Negatives are all other ground atoms of that predicate (closed world).
    """
    all_atoms = predicate_ground_atoms(constants, pred_name, arity)
    all_idx = [atom_to_idx[a] for a in all_atoms]

    pos_set = set(positive_atoms)
    pos_idx = [atom_to_idx[a] for a in all_atoms if a in pos_set]
    neg_idx = [atom_to_idx[a] for a in all_atoms if a not in pos_set]

    return Targets(all_idx=all_idx, pos_idx=pos_idx, neg_idx=neg_idx)

def predicate_ground_atoms_from_domains(
    pred_name: str,
    domains: List[List[str]],
) -> List[Atom]:
    """
    Generate all ground atoms pred_name(args...) using per-argument domains.
    domains length determines arity.
    """
    arity = len(domains)
    if arity == 0:
        return [Atom(pred_name, ())]
    if arity == 1:
        return [Atom(pred_name, (a0,)) for a0 in domains[0]]
    if arity == 2:
        return [Atom(pred_name, (a0, a1)) for a0 in domains[0] for a1 in domains[1]]
    if arity == 3:
        return [Atom(pred_name, (a0, a1, a2)) for a0 in domains[0] for a1 in domains[1] for a2 in domains[2]]
    raise ValueError(f"Unsupported arity for domains: {arity}")


def build_targets_from_positives_domains(
    *,
    atom_to_idx: Dict[Atom, int],
    pred_name: str,
    domains: List[List[str]],
    positive_atoms: List[Atom],
) -> "Targets":
    """
    Production version: builds targets using typed per-argument domains (not global constants).
    Negatives are all other ground atoms of pred_name over these domains (closed world on that domain).
    """
    all_atoms = predicate_ground_atoms_from_domains(pred_name, domains)
    all_idx = [atom_to_idx[a] for a in all_atoms]

    pos_set = set(positive_atoms)
    pos_idx = [atom_to_idx[a] for a in all_atoms if a in pos_set]
    neg_idx = [atom_to_idx[a] for a in all_atoms if a not in pos_set]

    return Targets(all_idx=all_idx, pos_idx=pos_idx, neg_idx=neg_idx)