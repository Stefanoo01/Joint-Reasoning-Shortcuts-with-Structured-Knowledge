from __future__ import annotations

from logic.atoms import Predicate
from logic.templates import RuleTemplate
from learning.bias import BiasConfig
from learning.task_config import TaskConfig


def make_config() -> TaskConfig:
    # Constants
    positions = ["d1", "d2"]
    digits = [str(d) for d in range(10)]
    sums = [str(s) for s in range(19)]

    # IMPORTANT: dedupe constants (0..9 appears in digits and sums)
    raw = positions + digits + sums
    C = list(dict.fromkeys(raw))  # preserves order

    predicates = [
        Predicate("digit", 2, "E"),     # digit(pos, d) from CBM (soft)
        Predicate("add", 3, "E"),       # add(A,B,S) truth table (hard)
        Predicate("tmp", 2, "I"),       # tmp(S,A) aux to make add expressible with 2-body clauses
        Predicate("sum_is", 1, "I"),    # target
    ]

    # Templates Π
    # tmp(S,A) :- digit(d2,B), add(A,B,S)  --> needs v=1 (B existential), int_flag=0
    tau_tmp = RuleTemplate(v=1, int_flag=0)

    # sum_is(S) :- digit(d1,A), tmp(S,A)  --> needs v=1 (A existential), int_flag=1 (uses tmp)
    tau_sum_1 = RuleTemplate(v=1, int_flag=0)  # allow some non-int base if needed
    tau_sum_2 = RuleTemplate(v=1, int_flag=1)  # allow intensional in body

    templates = {
        ("tmp", 2): (tau_tmp, tau_tmp),
        ("sum_is", 1): (tau_sum_1, tau_sum_2),
    }

    # Bias (keep it tight to avoid shortcuts)
    bias = BiasConfig(
        allowed_body_preds={
            ("tmp", 2): {"digit", "add"},
            ("sum_is", 1): {"digit", "tmp", "sum_is"},
        },
        require_recursive={},
        require_body_connected=True,
    )

    require_recursive_on_C2 = {
        # we do NOT force recursion for sum_is
        ("sum_is", 1): False,
        ("tmp", 2): False,
    }

    return TaskConfig(
        constants=C,
        predicates=predicates,
        target_key=("sum_is", 1),
        aux_keys=[("tmp", 2)],  # tmp is the aux predicate
        templates=templates,
        T=2,  # usually 2 steps enough: derive tmp then sum_is
        bias=bias,
        require_recursive_on_C2=require_recursive_on_C2,
    )