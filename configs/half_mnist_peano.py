from __future__ import annotations

from ilp.logic.atoms import Predicate
from ilp.logic.templates import RuleTemplate
from ilp.learning.bias import BiasConfig
from ilp.learning.task_config import TaskConfig
from ilp.logic.clauses import Clause, Var

def make_config() -> TaskConfig:
    """
    ILP config for HALFMNIST Peano Task.
    Valid sums are in 0..8. Digits are 0..4.
    """
    MODE = "medium"

    digits = [str(d) for d in range(5)]
    sums   = [str(s) for s in range(9)]
    valid_sums = ["0", "1", "5", "6"]

    raw = digits + sums
    C = list(dict.fromkeys(raw))

    arg_domains = {
        ("digit1", 1): [digits],
        ("digit2", 1): [digits],
        ("sum_is", 1): [sums],
        ("zero", 1):   [sums],
        ("succ", 2):   [sums, sums],
        ("eq", 2):     [sums, sums],
        ("tmp", 2):    [sums, digits],
        ("add_p2", 3): [digits, digits, sums],
        ("add_p", 3):  [digits, digits, sums],
    }

    predicates = [
        Predicate("digit1", 1, "E"),
        Predicate("digit2", 1, "E"),
        Predicate("zero",   1, "E"),
        Predicate("succ",   2, "E"),
        Predicate("eq",     2, "E"),
        Predicate("tmp",    2, "I"),
        Predicate("add_p2", 3, "I"),
        Predicate("add_p",  3, "I"),
        Predicate("sum_is", 1, "I"),
    ]

    # Templates
    # Base case uses eq, so it doesn't need existential vars (already covered by head vars)
    tau_add_base = RuleTemplate(v=0, int_flag=0)
    tau_add_rec  = RuleTemplate(v=1, int_flag=1)
    
    tau_add2     = RuleTemplate(v=1, int_flag=1)
    
    tau_tmp = RuleTemplate(v=1, int_flag=1)
    
    tau_sum_1 = RuleTemplate(v=1, int_flag=1)
    tau_sum_2 = RuleTemplate(v=1, int_flag=1)

    templates = {
        ("add_p2", 3): (tau_add2, tau_add2),
        ("add_p", 3):  (tau_add_base, tau_add_rec),
        ("tmp", 2):    (tau_tmp, tau_tmp),
        ("sum_is", 1): (tau_sum_1, tau_sum_2),
    }

    def tmp_mode_filter(c: Clause) -> bool:
        b1, b2 = c.body
        preds = {b1.pred, b2.pred}
        if preds != {"digit2", "add_p"}:
            return False

        head_x, head_y = c.head.args  # tmp(X,Y)
        digit_atom = b1 if b1.pred == "digit2" else b2
        add_atom   = b2 if digit_atom is b1 else b1

        (z0,) = digit_atom.args
        if not isinstance(z0, Var) or z0.name != "Z0":
            return False

        if MODE == "tight":
            allowed_add_args = {
                (head_y, z0, head_x),
                (z0, head_y, head_x),
            }
            return add_atom.args in allowed_add_args

        allowed_add_args = {
            (head_y, z0, head_x),
            (z0, head_y, head_x),
            (head_x, head_y, z0),
            (head_x, z0, head_y),
            (head_y, head_x, z0),
            (z0, head_x, head_y)
        }
        return add_atom.args in allowed_add_args

    def sum_mode_filter(c: Clause) -> bool:
        b1, b2 = c.body
        preds = {b1.pred, b2.pred}
        if preds != {"digit1", "tmp"}:
            return False

        (head_x,) = c.head.args  # sum_is(X)
        digit_atom = b1 if b1.pred == "digit1" else b2
        tmp_atom   = b2 if digit_atom is b1 else b1

        (z0,) = digit_atom.args
        if not isinstance(z0, Var) or z0.name != "Z0":
            return False

        if MODE == "tight":
            allowed_tmp_args = {
                (head_x, z0),
                (z0, head_x),
            }
            return tmp_atom.args in allowed_tmp_args

        allowed_tmp_args = {
            (head_x, z0),
            (z0, head_x),
            (head_x, head_x),
            (z0, z0)
        }
        return tmp_atom.args in allowed_tmp_args

    # Add_p / Add_p2 tight filters to prevent explosion of the search space with 3 var heads.
    # The required rule is: add_p(X,Y,Z) :- zero(Y), eq(X,Z)
    # The required rule is: add_p2(X,Y,Z) :- succ(Z0, Y), add_p(X, Z0, Z)
    # The required rule is: add_p(X,Y,Z) :- succ(Z0, Z), add_p2(X, Y, Z0)

    bias = BiasConfig(
        allowed_body_preds={
            ("add_p2", 3): {"succ", "add_p"},
            ("add_p", 3):  {"zero", "eq", "add_p2", "succ"},
            ("tmp", 2):    {"digit2", "add_p"},
            ("sum_is", 1): {"digit1", "tmp"},
        },
        require_recursive={},
        require_body_connected=True,
        custom_clause_filters={
            ("tmp", 2):    tmp_mode_filter,
            ("sum_is", 1): sum_mode_filter,
        },
    )

    require_recursive_on_C2 = {
        ("add_p2", 3): False,
        ("add_p", 3):  False,
        ("sum_is", 1): False,
        ("tmp", 2):    False,
    }

    return TaskConfig(
        constants=C,
        predicates=predicates,
        arg_domains=arg_domains,
        target_key=("sum_is", 1),
        aux_keys=[("add_p2", 3), ("add_p", 3), ("tmp", 2)],
        templates=templates,
        T=8,
        bias=bias,
        require_recursive_on_C2=require_recursive_on_C2,
    )
