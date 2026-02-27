from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from logic.atoms import Atom, BOT, Predicate


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    """
    Language specification:
      - constants: finite set of constants (as strings)
      - predicates: list of Predicate symbols (both E and I)
    """
    constants: List[str]
    predicates: List[Predicate]

    def __post_init__(self) -> None:
        if not self.constants:
            raise ValueError("LanguageSpec.constants must be non-empty")
        for c in self.constants:
            if not isinstance(c, str) or not c:
                raise ValueError("All constants must be non-empty strings")

        if not self.predicates:
            raise ValueError("LanguageSpec.predicates must be non-empty")

        for p in self.predicates:
            if p.name == BOT.pred:
                raise ValueError(f"Predicate name '{BOT.pred}' is reserved for BOT")

        seen = set()
        for p in self.predicates:
            key = (p.name, p.arity)
            if key in seen:
                raise ValueError(f"Duplicate predicate symbol: {p.name}/{p.arity}")
            seen.add(key)

def build_ground_atoms(spec: LanguageSpec) -> list[Atom]:
    """
    Build the ordered list G of all ground atoms over the language,
    plus BOT as the last atom.
    """

    sorted_cns = sorted(spec.constants)
    sorted_preds = sorted(
        spec.predicates,
        key=lambda p: (0 if p.kind == "E" else 1, p.name, p.arity),
    )

    G: List[Atom] = []

    for p in sorted_preds:
        if p.arity == 0:
            G.append(Atom(p.name, ()))

        elif p.arity == 1:
            for c in sorted_cns:
                G.append(Atom(p.name, (c,)))

        elif p.arity == 2:
            for c1 in sorted_cns:
                for c2 in sorted_cns:
                    G.append(Atom(p.name, (c1, c2)))

        elif p.arity == 3:
            for c1 in sorted_cns:
                for c2 in sorted_cns:
                    for c3 in sorted_cns:
                        G.append(Atom(p.name, (c1, c2, c3)))

        else:
            raise ValueError(f"Unsupported arity: {p.arity}")

    G.append(BOT)

    # Check duplicates
    if len(set(G)) != len(G):
        raise ValueError("Duplicate ground atoms generated (check predicates/constants)")

    return G

def build_index(G: List[Atom]) -> Tuple[Dict[Atom, int], List[Atom], int]:
    """
    Create stable index mappings for atoms in G.
    """
    idx_to_atom = list(G)
    atom_to_idx: Dict[Atom, int] = {a: i for i, a in enumerate(idx_to_atom)}
    bot_idx = atom_to_idx[BOT]

    return atom_to_idx, idx_to_atom, bot_idx