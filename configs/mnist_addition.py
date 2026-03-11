from __future__ import annotations

from ilp.logic.atoms import Predicate
from ilp.logic.templates import RuleTemplate
from ilp.learning.bias import BiasConfig
from ilp.learning.task_config import TaskConfig
from ilp.learning.data import PredicateKey

# Import types only for type hints (not required at runtime)
from ilp.logic.clauses import Clause
from ilp.logic.clauses import Var


def make_config() -> TaskConfig:
    MODE = "medium"   # "tight" or "medium"
    # Constants (typed domains)
    digits = [str(d) for d in range(10)]
    sums = [str(s) for s in range(19)]

    # IMPORTANT: dedupe constants (0..9 appears in digits and sums)
    raw = digits + sums
    C = list(dict.fromkeys(raw))

    # Typed grounding domains per predicate/arg
    arg_domains = {
        ("digit1", 1): [digits],
        ("digit2", 1): [digits],
        ("sum_is", 1): [sums],
        ("tmp", 2): [sums, digits],      # tmp(S, A)
        ("add", 3): [digits, digits, sums],  # add(A, B, S)
    }

    predicates = [
        Predicate("digit1", 1, "E"),   # from CBM image1
        Predicate("digit2", 1, "E"),   # from CBM image2
        Predicate("add", 3, "E"),      # truth table hard facts
        Predicate("tmp", 2, "I"),      # tmp(S,A) aux
        Predicate("sum_is", 1, "I"),   # target
    ]

    # Templates Π
    # tmp(S,A) :- digit2(B), add(A,B,S)  needs v=1 (B), int_flag=0
    tau_tmp = RuleTemplate(v=1, int_flag=0)

    # sum_is(S) :- digit1(A), tmp(S,A)  needs v=1 (A), int_flag=1 (uses intensional tmp)
    tau_sum_1 = RuleTemplate(v=1, int_flag=1)
    tau_sum_2 = RuleTemplate(v=1, int_flag=1)

    templates = {
        ("tmp", 2): (tau_tmp, tau_tmp),
        ("sum_is", 1): (tau_sum_1, tau_sum_2),
    }

    # --------- Custom "mode" filters (HUGE pruning) ---------
    # Vars are created by your generator:
    # - head tmp/2 uses Var("X"), Var("Y")
    # - v=1 introduces Var("Z0")
    #
    # We enforce exactly:
    #   tmp(X,Y) :- digit2(Z0), add(Y,Z0,X).
    #
    # and for sum_is/1:
    #   sum_is(X) :- digit1(Z0), tmp(X,Z0).
    #
    # This collapses tmp clause sets from hundreds to ~1.

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

        # ---- TIGHT: only add(Y, Z0, X) ----
        if MODE == "tight":
            return add_atom.args == (head_y, z0, head_x)

        # ---- MEDIUM: allow a few variants (still tiny search space) ----
        allowed_add_args = {
            (head_y, z0, head_x),   # correct: add(A,B,S) with tmp(S,A)
            (z0, head_y, head_x),   # swap A/B (should still be valid because add is commutative in truth table)
        }
        return add_atom.args in allowed_add_args


    def sum_mode_filter(c: Clause) -> bool:
        b1, b2 = c.body
        preds = {b1.pred, b2.pred}
        if preds != {"digit1", "tmp"}:
            return False

        (head_x,) = c.head.args  # sum_is(X)

        digit_atom = b1 if b1.pred == "digit1" else b2
        tmp_atom = b2 if digit_atom is b1 else b1

        (z0,) = digit_atom.args
        if not isinstance(z0, Var) or z0.name != "Z0":
            return False

        # ---- TIGHT: only tmp(X, Z0) ----
        if MODE == "tight":
            return tmp_atom.args == (head_x, z0)

        # ---- MEDIUM: allow a couple variants ----
        allowed_tmp_args = {
            (head_x, z0),  # intended: tmp(S,A) with A from digit1
            (z0, head_x),  # “wrong” wiring (lets ILP try a shortcut / failure mode)
        }
        return tmp_atom.args in allowed_tmp_args

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