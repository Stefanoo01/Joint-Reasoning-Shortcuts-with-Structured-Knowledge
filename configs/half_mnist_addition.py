from __future__ import annotations

from logic.atoms import Predicate
from logic.templates import RuleTemplate
from learning.bias import BiasConfig
from learning.task_config import TaskConfig
from learning.data import PredicateKey

# Import types only for type hints (not required at runtime)
from logic.clauses import Clause
from logic.clauses import Var


def make_config() -> TaskConfig:
    """
    ILP config for the HALFMNIST dataset.

    HALFMNIST restricts digits to {0,1,2,3,4} and, after the `filtrate`
    function in halfmnist.py, only the following digit pairs survive:
        (0,0), (0,1), (1,0), (2,3), (2,4), (3,2), (4,2)
    yielding valid sums: {0, 1, 5, 6}.

    The add/3 truth table still covers *all* (A, B, S) with A,B in 0..4,
    but the target predicate sum_is/1 is grounded only over {0,1,5,6}.
    """
    MODE = "medium"  # "tight" or "medium"

    # Constants (typed domains) ------------------------------------------------
    digits = [str(d) for d in range(5)]                # 0..4
    sums   = [str(s) for s in range(9)]                # 0..8 (all reachable by add)
    valid_sums = ["0", "1", "5", "6"]                  # the 4 target classes

    # Dedupe constants (0..4 appears in both digits and sums)
    raw = digits + sums
    C = list(dict.fromkeys(raw))

    # Typed grounding domains per predicate/arg --------------------------------
    arg_domains = {
        ("digit1", 1): [digits],
        ("digit2", 1): [digits],
        ("sum_is", 1): [sums],                         # ground over all reachable sums
        ("tmp", 2):    [sums, digits],                 # tmp(S, A)
        ("add", 3):    [digits, digits, sums],         # add(A, B, S)
    }

    predicates = [
        Predicate("digit1", 1, "E"),   # from CBM image1
        Predicate("digit2", 1, "E"),   # from CBM image2
        Predicate("add",    3, "E"),   # truth table hard facts
        Predicate("tmp",    2, "I"),   # tmp(S,A) aux
        Predicate("sum_is", 1, "I"),   # target
    ]

    # Templates Π ---------------------------------------------------------------
    # tmp(S,A) :- digit2(B), add(A,B,S)   needs v=1 (B), int_flag=0
    tau_tmp = RuleTemplate(v=1, int_flag=0)

    # sum_is(S) :- digit1(A), tmp(S,A)    needs v=1 (A), int_flag=1
    tau_sum_1 = RuleTemplate(v=1, int_flag=1)
    tau_sum_2 = RuleTemplate(v=1, int_flag=1)

    templates = {
        ("tmp", 2):    (tau_tmp, tau_tmp),
        ("sum_is", 1): (tau_sum_1, tau_sum_2),
    }

    # --------- Custom "mode" filters (HUGE pruning) ---------------------------
    def tmp_mode_filter(c: Clause) -> bool:
        b1, b2 = c.body
        preds = {b1.pred, b2.pred}
        if preds != {"digit2", "add"}:
            return False

        head_x, head_y = c.head.args  # tmp(X,Y)
        digit_atom = b1 if b1.pred == "digit2" else b2
        add_atom   = b2 if digit_atom is b1 else b1

        (z0,) = digit_atom.args
        if not isinstance(z0, Var) or z0.name != "Z0":
            return False

        if MODE == "tight":
            return add_atom.args == (head_y, z0, head_x)

        allowed_add_args = {
            (head_y, z0, head_x),   # correct: add(A,B,S) with tmp(S,A)
            (z0, head_y, head_x),   # swap A/B  (commutative)
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
            return tmp_atom.args == (head_x, z0)

        allowed_tmp_args = {
            (head_x, z0),  # intended: tmp(S,A) with A from digit1
            (z0, head_x),  # "wrong" wiring (lets ILP try a shortcut)
        }
        return tmp_atom.args in allowed_tmp_args

    bias = BiasConfig(
        allowed_body_preds={
            ("tmp", 2):    {"digit2", "add"},
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
        ("sum_is", 1): False,
        ("tmp", 2):    False,
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
