from __future__ import annotations

from ilp.learning.task_config import TaskConfig
from ilp.learning.bias import BiasConfig
from ilp.logic.atoms import Predicate
from ilp.logic.templates import RuleTemplate

from ilp.logic.clauses import Clause, Var  # usa i tuoi tipi


def make_config() -> TaskConfig:
    digits = [str(d) for d in range(5)]      # 0..4
    sums = [str(s) for s in range(9)]        # 0..8

    # Dedup constants
    raw = digits + sums
    C = list(dict.fromkeys(raw))

    arg_domains = {
        ("digit1", 1): [digits],
        ("digit2", 1): [digits],
        ("sum_is", 1): [sums],
        ("tmp", 2): [sums, digits],              # tmp(S,A)
        ("add", 3): [digits, digits, sums],      # add(A,B,S)
    }

    predicates = [
        Predicate("digit1", 1, "E"),
        Predicate("digit2", 1, "E"),
        Predicate("add", 3, "E"),
        Predicate("tmp", 2, "I"),
        Predicate("sum_is", 1, "I"),
    ]

    tau_tmp = RuleTemplate(v=1, int_flag=0)
    tau_sum_1 = RuleTemplate(v=1, int_flag=1)
    tau_sum_2 = RuleTemplate(v=1, int_flag=1)

    templates = {
        ("tmp", 2): (tau_tmp, tau_tmp),
        ("sum_is", 1): (tau_sum_1, tau_sum_2),
    }

    # Bias “medium safe”: tmp ha 2 varianti (commutatività A,B),
    # sum_is resta “safe” (1 sola) per stabilità.
    MODE = "medium"

    def tmp_mode_filter(c: Clause) -> bool:
        b1, b2 = c.body
        preds = {b1.pred, b2.pred}
        if preds != {"digit2", "add"}:
            return False

        head_x, head_y = c.head.args  # tmp(X,Y)
        digit_atom = b1 if b1.pred == "digit2" else b2
        add_atom = b2 if digit_atom is b1 else b1

        (z0,) = digit_atom.args
        if not isinstance(z0, Var) or z0.name != "Z0":
            return False

        if MODE == "tight":
            return add_atom.args == (head_y, z0, head_x)

        allowed_add_args = {
            (head_y, z0, head_x),
            (z0, head_y, head_x),
        }
        return add_atom.args in allowed_add_args

    def sum_mode_filter(c: Clause) -> bool:
        b1, b2 = c.body
        preds = {b1.pred, b2.pred}
        if preds != {"digit1", "tmp"}:
            return False

        (head_x,) = c.head.args
        digit_atom = b1 if b1.pred == "digit1" else b2
        tmp_atom = b2 if digit_atom is b1 else b1

        (z0,) = digit_atom.args
        if not isinstance(z0, Var) or z0.name != "Z0":
            return False

        # safe wiring only
        return tmp_atom.args == (head_x, z0)

    bias = BiasConfig(
        allowed_body_preds={
            ("tmp", 2): {"digit2", "add"},
            ("sum_is", 1): {"digit1", "tmp"},
        },
        require_recursive={},
        require_body_connected=True,
        custom_clause_filters={
            ("tmp", 2): tmp_mode_filter,
            ("sum_is", 1): sum_mode_filter,
        },
    )

    require_recursive_on_C2 = {
        ("sum_is", 1): False,
        ("tmp", 2): False,
    }

    return TaskConfig(
        constants=C,
        predicates=predicates,
        arg_domains=arg_domains,
        target_key=("sum_is", 1),
        aux_keys=[("tmp", 2)],
        templates=templates,
        T=2,
        bias=bias,
        require_recursive_on_C2=require_recursive_on_C2,
    )