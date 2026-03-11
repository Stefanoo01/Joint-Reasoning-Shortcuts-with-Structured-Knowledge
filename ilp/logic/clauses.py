from __future__ import annotations
from dataclasses import dataclass
from typing import Set, Tuple
from ilp.logic.atoms import Predicate
from ilp.logic.templates import RuleTemplate

@dataclass(frozen=True, slots=True)
class Var:
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Var.name must be a non-empty string")

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class AtomSchema:
    """
    Non-ground atom with variables only, e.g. succ(X,Z)
    """
    pred: str
    args: Tuple[Var, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pred, str) or not self.pred:
            raise ValueError("AtomSchema.pred must be a non-empty string")
        if not isinstance(self.args, tuple):
            raise ValueError("AtomSchema.args must be a tuple[Var,...]")
        for v in self.args:
            if not isinstance(v, Var):
                raise ValueError("AtomSchema.args must contain only Var")

    @property
    def arity(self) -> int:
        return len(self.args)

    def vars(self) -> Set[Var]:
        return set(self.args)

    def __str__(self) -> str:
        if self.arity == 0:
            return self.pred
        return f"{self.pred}({','.join(str(v) for v in self.args)})"


@dataclass(frozen=True, slots=True)
class Clause:
    """
    Definite clause with exactly 2 body atoms:
      head :- body1, body2.
    """
    head: AtomSchema
    body: Tuple[AtomSchema, AtomSchema]

    def __post_init__(self) -> None:
        if not isinstance(self.body, tuple) or len(self.body) != 2:
            raise ValueError("Clause.body must be a tuple of exactly 2 AtomSchema")
        for b in self.body:
            if not isinstance(b, AtomSchema):
                raise ValueError("Clause.body must contain AtomSchema")

    def head_vars(self) -> Set[Var]:
        return self.head.vars()

    def body_vars(self) -> Set[Var]:
        return self.body[0].vars().union(self.body[1].vars())

    def is_safe(self) -> bool:
        # every head var must appear in body
        return self.head_vars().issubset(self.body_vars())

    def is_circular(self) -> bool:
        # head atom appears syntactically in the body
        return self.body[0] == self.head or self.body[1] == self.head

    def canonical_key(self) -> tuple:
        """
        Canonical key for duplicate pruning under body atom swap.
        We ignore body order by sorting their string representations.
        """
        head_key = str(self.head)
        b1, b2 = str(self.body[0]), str(self.body[1])
        body_key = tuple(sorted([b1, b2]))
        return (head_key, body_key)

    def __str__(self) -> str:
        return f"{self.head} :- {self.body[0]}, {self.body[1]}."

    def is_body_connected(self) -> bool:
        b1_vars = self.body[0].vars()
        b2_vars = self.body[1].vars()
        return len(b1_vars.intersection(b2_vars)) > 0

def make_vars_for_head(head_arity: int, v: int) -> tuple[list[Var], list[Var]]:
    if head_arity not in {0, 1, 2, 3}:
        raise ValueError("head_arity must be 0, 1, 2, or 3")
    if v < 0:
        raise ValueError("v must be >= 0")

    head_vars: list[Var] = []
    if head_arity >= 1:
        head_vars.append(Var("X"))
    if head_arity >= 2:
        head_vars.append(Var("Y"))
    if head_arity >= 3:
        head_vars.append(Var("Z"))

    extra_vars = [Var(f"Z{i}") for i in range(v)]
    return head_vars, head_vars + extra_vars


def allowed_body_predicates(predicates: list[Predicate], int_flag: int) -> list[Predicate]:
    if int_flag not in {0, 1}:
        raise ValueError("int_flag must be 0 or 1")
    if int_flag == 0:
        return [p for p in predicates if p.kind == "E"]
    return list(predicates)


def enumerate_atom_schemas(pred: Predicate, vars_all: list[Var]) -> list[AtomSchema]:
    if pred.arity == 0:
        return [AtomSchema(pred.name, ())]
    if pred.arity == 1:
        return [AtomSchema(pred.name, (v,)) for v in vars_all]
    if pred.arity == 2:
        out: list[AtomSchema] = []
        for v1 in vars_all:
            for v2 in vars_all:
                out.append(AtomSchema(pred.name, (v1, v2)))
        return out
    if pred.arity == 3:
        out: list[AtomSchema] = []
        for v1 in vars_all:
            for v2 in vars_all:
                for v3 in vars_all:
                    out.append(AtomSchema(pred.name, (v1, v2, v3)))
        return out
    raise ValueError("Predicate arity must be 0,1,2")


def generate_body_atom_candidates(
    head_pred: Predicate,
    predicates: list[Predicate],
    template: RuleTemplate,
) -> tuple[list[Var], list[Var], list[AtomSchema]]:
    """
    Returns:
      - head_vars: variables used in head (e.g., [X] or [X,Y])
      - vars_all: all variables used in body
      - body_atoms: all AtomSchema candidates allowed in the body under template
    """
    head_vars, vars_all = make_vars_for_head(head_pred.arity, template.v)
    allowed_preds = allowed_body_predicates(predicates, template.int_flag)

    body_atoms: list[AtomSchema] = []
    for p in allowed_preds:
        body_atoms.extend(enumerate_atom_schemas(p, vars_all))

    return head_vars, vars_all, body_atoms

def make_head_schema(head_pred: Predicate, head_vars: list[Var]) -> AtomSchema:
    if head_pred.arity == 0:
        return AtomSchema(head_pred.name, ())
    if head_pred.arity == 1:
        return AtomSchema(head_pred.name, (head_vars[0],))
    if head_pred.arity == 2:
        return AtomSchema(head_pred.name, (head_vars[0], head_vars[1]))
    if head_pred.arity == 3:
        return AtomSchema(head_pred.name, (head_vars[0], head_vars[1], head_vars[2]))
    raise ValueError("head_pred.arity must be 0,1,2")

def _var_bit_index(vars_all: list[Var]) -> dict[Var, int]:
    return {v: i for i, v in enumerate(vars_all)}

def _atom_mask(atom: AtomSchema, bit_index: dict[Var, int]) -> int:
    m = 0
    for v in atom.args:
        m |= 1 << bit_index[v]
    return m

def generate_clauses_for_template(
    head_pred: Predicate,
    predicates: list[Predicate],
    template: RuleTemplate,
) -> list[Clause]:
    head_vars, vars_all, body_atoms = generate_body_atom_candidates(head_pred, predicates, template)
    head = make_head_schema(head_pred, head_vars)

    # Pre-prune circular: remove atoms identical to head before pairing
    body_atoms = [b for b in body_atoms if b != head]

    # Build bitmasks once
    bit_index = _var_bit_index(vars_all)
    head_mask = _atom_mask(head, bit_index)
    masks = [_atom_mask(b, bit_index) for b in body_atoms]

    out: list[Clause] = []
    n = len(body_atoms)

    # Duplicate pruning under swap: only consider j >= i
    for i in range(n):
        b1 = body_atoms[i]
        m1 = masks[i]
        for j in range(i, n):
            m2 = masks[j]

            # Safety pruning: head vars must appear in the body
            if ((m1 | m2) & head_mask) != head_mask:
                continue

            # Connectedness pruning: head + body atoms must form a connected
            # variable graph.  Two body atoms are "connected" if they share
            # a variable directly (m1 & m2 != 0) **or** if both share at
            # least one variable with the head.
            if (m1 & m2) == 0:
                # Body atoms don't share a variable directly — check that
                # each body atom is connected to the head instead.
                if (m1 & head_mask) == 0 or (m2 & head_mask) == 0:
                    continue

            out.append(Clause(head=head, body=(b1, body_atoms[j])))

    return out