from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Predicate:
    """
    Predicate symbol with arity and kind.
    kind:
      - "E" = extensional (facts in B)
      - "I" = intensional (learned/derived, incl. target and auxiliaries)
    """
    name: str
    arity: int
    kind: str  # "E" or "I"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Predicate.name must be a non-empty string")
        if self.kind not in {"E", "I"}:
            raise ValueError("Predicate.kind must be 'E' or 'I'")
        if self.arity < 0 or self.arity > 3:
            raise ValueError("Predicate.arity must be 0, 1, 2, or 3")

@dataclass(frozen=True, slots=True)
class Atom:
    """
    Ground atom (predicate applied to constants).
    Examples:
      Atom("p", ())          represents p/0
      Atom("q", ("a",))      represents q(a)
      Atom("r", ("a","b"))   represents r(a,b)
    """
    pred: str
    args: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pred, str) or not self.pred:
            raise ValueError("Atom.pred must be a non-empty string")
        if not isinstance(self.args, tuple):
            raise ValueError("Atom.args must be a tuple[str, ...] (not a list)")
        if not all(isinstance(a, str) for a in self.args):
            raise ValueError("All Atom.args must be strings")

    @property
    def arity(self) -> int:
        return len(self.args)

    def __str__(self) -> str:
        if self.arity == 0:
            return self.pred
        return f"{self.pred}({','.join(self.args)})"


# Constant: falsum / bottom atom ⊥
BOT = Atom("__bot__", ())