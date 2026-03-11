from __future__ import annotations

from logic.atoms import Predicate
from logic.templates import RuleTemplate
from learning.bias import BiasConfig
from learning.task_config import TaskConfig
from logic.clauses import Clause


def make_config(mode: str = "relaxed") -> TaskConfig:
    """
    ILP config for the toy sum-parity task.

    Two modes:
      - "relaxed": fully open bias, no custom filters.
                   Produces the reasoning shortcut: sum_even(X) :- digit1(X), digit2(X).
      - "guided":  minimal hierarchical constraints that lead to the correct program.
                   pair_same restricted to parity predicates, check/sum_even must
                   use the invented predicate from the level below + a digit.
    """
    digits = [str(d) for d in range(5)]  # "0".."4"

    arg_domains = {
        ("digit1", 1):     [digits],
        ("digit2", 1):     [digits],
        ("is_even", 1):    [digits],
        ("is_odd", 1):     [digits],
        ("pair_same", 2):  [digits, digits],
        ("check", 2):      [digits, digits],
        ("sum_even", 1):   [digits],
    }

    predicates = [
        Predicate("digit1", 1, "E"),
        Predicate("digit2", 1, "E"),
        Predicate("is_even", 1, "E"),
        Predicate("is_odd", 1, "E"),
        Predicate("pair_same", 2, "I"),
        Predicate("check", 2, "I"),
        Predicate("sum_even", 1, "I"),
    ]

    tau_pair_same = RuleTemplate(v=0, int_flag=0)
    tau_check     = RuleTemplate(v=0, int_flag=1)
    tau_sum_even  = RuleTemplate(v=1, int_flag=1)

    templates = {
        ("pair_same", 2): (tau_pair_same, tau_pair_same),
        ("check", 2):     (tau_check, tau_check),
        ("sum_even", 1):  (tau_sum_even, tau_sum_even),
    }

    if mode == "relaxed":
        # ----- MODE 1: fully open bias -----
        # No custom filters. The ILP is free to compose any predicate at any
        # level. This reliably produces a reasoning shortcut (~91.7% acc).
        bias = BiasConfig(
            allowed_body_preds={
                ("pair_same", 2): {"digit1", "digit2", "is_even", "is_odd"},
                ("check", 2):     {"digit1", "digit2", "is_even", "is_odd", "pair_same"},
                ("sum_even", 1):  {"digit1", "digit2", "is_even", "is_odd", "pair_same", "check"},
            },
            require_recursive={},
            require_body_connected=False,
        )

    elif mode == "guided":
        # ----- MODE 2: minimal guided bias -----
        # pair_same can ONLY use parity predicates (is_even, is_odd).
        # check MUST use pair_same + a digit predicate.
        # sum_even MUST use check + a digit predicate.
        # Variable wiring is completely free.

        def check_filter(c: Clause) -> bool:
            preds = {b.pred for b in c.body}
            return "pair_same" in preds and ("digit1" in preds or "digit2" in preds)

        def sum_even_filter(c: Clause) -> bool:
            preds = {b.pred for b in c.body}
            return "check" in preds and ("digit1" in preds or "digit2" in preds)

        bias = BiasConfig(
            allowed_body_preds={
                ("pair_same", 2): {"is_even", "is_odd"},
                ("check", 2):     {"digit1", "digit2", "pair_same"},
                ("sum_even", 1):  {"digit1", "digit2", "check"},
            },
            require_recursive={},
            require_body_connected=False,
            custom_clause_filters={
                ("check", 2):     check_filter,
                ("sum_even", 1):  sum_even_filter,
            },
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'relaxed' or 'guided'.")

    require_recursive_on_C2 = {
        ("pair_same", 2): False,
        ("check", 2):     False,
        ("sum_even", 1):  False,
    }

    return TaskConfig(
        constants=digits,
        predicates=predicates,
        arg_domains=arg_domains,
        target_key=("sum_even", 1),
        aux_keys=[("pair_same", 2), ("check", 2)],
        templates=templates,
        T=3,
        bias=bias,
        require_recursive_on_C2=require_recursive_on_C2,
    )
