from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from ilp.logic.atoms import Predicate

@dataclass(frozen=True, slots=True)
class RuleTemplate:
    """
    Rule template τ = (v, int_flag)
      - v: number of existential variables allowed in the body
      - int_flag: 0 => body uses only extensional predicates
                  1 => body may use intensional predicates too
    """
    v: int
    int_flag: int  # 0 or 1

    def __post_init__(self) -> None:
        if not isinstance(self.v, int) or self.v < 0:
            raise ValueError("RuleTemplate.v must be an int >= 0")
        if self.int_flag not in {0, 1}:
            raise ValueError("RuleTemplate.int_flag must be 0 or 1")

PredId = Tuple[str, int]  # (name, arity)

@dataclass(frozen=True, slots=True)
class ProgramTemplate:
    """
    Program template Π = (P_a, arity_a, rules, T)
    We encode:
      - P_a via aux_predicates (each has name + arity, kind='I')
      - arity_a is implicit in Predicate.arity
      - rules maps each intensional predicate id to a pair of RuleTemplate
      - T is max #inference steps
    """
    aux_predicates: List[Predicate]
    rules: Dict[PredId, Tuple[RuleTemplate, RuleTemplate]]
    T: int

    def __post_init__(self) -> None:
        if not isinstance(self.T, int) or self.T <= 0:
            raise ValueError("ProgramTemplate.T must be an int > 0")

        # aux predicates must be intensional and unique
        seen: set[PredId] = set()
        for p in self.aux_predicates:
            if p.kind != "I":
                raise ValueError("All aux_predicates must have kind='I'")
            key = (p.name, p.arity)
            if key in seen:
                raise ValueError(f"Duplicate aux predicate: {p.name}/{p.arity}")
            seen.add(key)

        # rules keys must be well-formed and each value must be 2 templates
        for (name, arity), pair in self.rules.items():
            if not isinstance(name, str) or not name:
                raise ValueError("rules keys must have a non-empty predicate name")
            if arity not in {0, 1, 2, 3}:
                raise ValueError("rules keys must have arity 0,1,2,3")
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("rules[p] must be a pair (tau1, tau2)")

    @property
    def aux_set(self) -> set[PredId]:
        return {(p.name, p.arity) for p in self.aux_predicates}

    def intensional_predicates(self, target: Predicate) -> List[Predicate]:
        """
        Return P_i = P_a ∪ {target}
        """
        # Ensure target is intensional by convention
        if target.kind != "I":
            raise ValueError("target must have kind='I'")
        # Avoid duplicates if user also included target in aux_predicates
        out = list(self.aux_predicates)
        if (target.name, target.arity) not in self.aux_set:
            out.append(target)
        return out