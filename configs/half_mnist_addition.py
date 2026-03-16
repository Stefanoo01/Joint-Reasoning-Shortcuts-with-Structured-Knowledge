from __future__ import annotations

from dataclasses import dataclass

from ilp.logic.atoms import Predicate
from ilp.logic.templates import RuleTemplate
from ilp.learning.bias import BiasConfig
from ilp.learning.task_config import TaskConfig
from ilp.learning.data import PredicateKey

# Import types only for type hints (not required at runtime)
from ilp.logic.clauses import Clause
from ilp.logic.clauses import Var


@dataclass(frozen=True)
class _AdditionVariantSpec:
    name: str
    allowed_body_preds: dict[PredicateKey, set[str]]
    tmp_filter_kind: str
    sum_filter_kind: str
    add_extra_tmp2: bool
    tmp2_filter_kind: str | None
    require_recursive_on_C2: dict[PredicateKey, bool]


def _make_addition_variant_spec(variant: str) -> _AdditionVariantSpec:
    if variant == "base":
        return _AdditionVariantSpec(
            name="base",
            allowed_body_preds={
                ("tmp", 2): {"digit2", "add"},
                ("sum_is", 1): {"digit1", "tmp"},
            },
            tmp_filter_kind="mode",
            sum_filter_kind="mode",
            add_extra_tmp2=False,
            tmp2_filter_kind=None,
            require_recursive_on_C2={
                ("sum_is", 1): False,
                ("tmp", 2): False,
            },
        )

    if variant == "canonical_only":
        return _AdditionVariantSpec(
            name="canonical_only",
            allowed_body_preds={
                ("tmp", 2): {"digit2", "add"},
                ("sum_is", 1): {"digit1", "tmp"},
            },
            tmp_filter_kind="canonical_only",
            sum_filter_kind="canonical_only",
            add_extra_tmp2=False,
            tmp2_filter_kind=None,
            require_recursive_on_C2={
                ("sum_is", 1): False,
                ("tmp", 2): False,
            },
        )

    if variant == "broad_search":
        return _AdditionVariantSpec(
            name="broad_search",
            allowed_body_preds={
                ("tmp", 2): {"digit1", "digit2", "add"},
                ("sum_is", 1): {"digit1", "digit2", "tmp", "add"},
            },
            tmp_filter_kind="broad_structured",
            sum_filter_kind="broad_structured",
            add_extra_tmp2=False,
            tmp2_filter_kind=None,
            require_recursive_on_C2={
                ("sum_is", 1): False,
                ("tmp", 2): False,
            },
        )

    if variant == "sum_relaxed":
        return _AdditionVariantSpec(
            name="sum_relaxed",
            allowed_body_preds={
                ("tmp", 2): {"digit2", "add"},
                ("sum_is", 1): {"digit1", "digit2", "tmp"},
            },
            tmp_filter_kind="mode",
            sum_filter_kind="sum_relaxed",
            add_extra_tmp2=False,
            tmp2_filter_kind=None,
            require_recursive_on_C2={
                ("sum_is", 1): False,
                ("tmp", 2): False,
            },
        )

    if variant == "extra_tmp2":
        return _AdditionVariantSpec(
            name="extra_tmp2",
            allowed_body_preds={
                ("tmp", 2): {"digit1", "digit2", "add"},
                ("tmp2", 2): {"digit1", "digit2", "add"},
                ("sum_is", 1): {"digit1", "digit2", "tmp", "tmp2", "add"},
            },
            tmp_filter_kind="none",
            sum_filter_kind="none",
            add_extra_tmp2=True,
            tmp2_filter_kind="none",
            require_recursive_on_C2={
                ("sum_is", 1): False,
                ("tmp", 2): False,
                ("tmp2", 2): False,
            },
        )

    raise ValueError(
        "Unsupported HalfMNIST addition variant: "
        f"{variant}. Known variants: base, canonical_only, broad_search, sum_relaxed, extra_tmp2"
    )


def make_config(mode: str = "medium", T: int = 2, variant: str = "base") -> TaskConfig:
    """
    ILP config for the HALFMNIST dataset.

    HALFMNIST restricts digits to {0,1,2,3,4} and, after the `filtrate`
    function in halfmnist.py, only the following digit pairs survive:
        (0,0), (0,1), (1,0), (2,3), (2,4), (3,2), (4,2)
    yielding valid sums: {0, 1, 5, 6}.

    The add/3 truth table still covers *all* (A, B, S) with A,B in 0..4,
    but the target predicate sum_is/1 is grounded only over {0,1,5,6}.
    """
    if mode not in {"tight", "medium"}:
        raise ValueError(f"Unsupported HalfMNIST addition mode: {mode}")
    if T <= 0:
        raise ValueError("T must be > 0")

    MODE = mode
    variant_spec = _make_addition_variant_spec(variant)

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
    if variant_spec.add_extra_tmp2:
        arg_domains[("tmp2", 2)] = [sums, digits]

    predicates = [
        Predicate("digit1", 1, "E"),   # from CBM image1
        Predicate("digit2", 1, "E"),   # from CBM image2
        Predicate("add",    3, "E"),   # truth table hard facts
        Predicate("tmp",    2, "I"),   # tmp(S,A) aux
        Predicate("sum_is", 1, "I"),   # target
    ]
    if variant_spec.add_extra_tmp2:
        predicates.insert(4, Predicate("tmp2", 2, "I"))  # extra aux predicate

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
    if variant_spec.add_extra_tmp2:
        templates[("tmp2", 2)] = (tau_tmp, tau_tmp)

    # --------- Custom "mode" filters (HUGE pruning) ---------------------------
    def tmp_like_filter(c: Clause, filter_kind: str) -> bool:
        b1, b2 = c.body
        if filter_kind == "none":
            return True

        preds = {b1.pred, b2.pred}
        if "add" not in preds:
            return False
        digit_preds = preds.intersection({"digit1", "digit2"})
        if len(digit_preds) != 1:
            return False

        head_x, head_y = c.head.args  # tmp(X,Y)
        digit_atom = b1 if b1.pred in {"digit1", "digit2"} else b2
        add_atom   = b2 if digit_atom is b1 else b1

        (digit_var,) = digit_atom.args
        if not isinstance(digit_var, Var):
            return False

        if filter_kind == "broad_structured":
            return digit_var in add_atom.args

        z0 = digit_var
        if z0.name != "Z0":
            return False

        if filter_kind == "canonical_only":
            return add_atom.args == (head_y, z0, head_x)

        if MODE == "tight":
            allowed_add_args = {
                (head_y, z0, head_x),   # add(A,B,S)
                (z0, head_y, head_x),   # add(B,A,S)
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

    def tmp_mode_filter(c: Clause) -> bool:
        return tmp_like_filter(c, variant_spec.tmp_filter_kind)

    def tmp2_mode_filter(c: Clause) -> bool:
        if variant_spec.tmp2_filter_kind is None:
            raise RuntimeError("tmp2 filter requested without tmp2 variant enabled")
        return tmp_like_filter(c, variant_spec.tmp2_filter_kind)

    def sum_mode_filter(c: Clause) -> bool:
        if variant_spec.sum_filter_kind == "none":
            return True

        b1, b2 = c.body
        preds = {b1.pred, b2.pred}
        if variant_spec.sum_filter_kind == "broad_structured":
            allowed_pred_pairs = {
                frozenset({"digit1", "tmp"}),
                frozenset({"digit2", "tmp"}),
                frozenset({"digit1", "add"}),
                frozenset({"digit2", "add"}),
            }
            if frozenset(preds) not in allowed_pred_pairs:
                return False

            (head_x,) = c.head.args
            digit_atom = b1 if b1.pred in {"digit1", "digit2"} else b2
            rel_atom = b2 if digit_atom is b1 else b1

            (digit_var,) = digit_atom.args
            if not isinstance(digit_var, Var):
                return False

            return digit_var in rel_atom.args and head_x in rel_atom.args

        if variant_spec.sum_filter_kind == "sum_relaxed":
            if preds not in ({"digit1", "tmp"}, {"digit2", "tmp"}):
                return False

            (head_x,) = c.head.args
            digit_atom = b1 if b1.pred in {"digit1", "digit2"} else b2
            tmp_atom = b2 if digit_atom is b1 else b1

            (digit_var,) = digit_atom.args
            if not isinstance(digit_var, Var):
                return False

            return digit_var in tmp_atom.args and head_x in tmp_atom.args

        if preds != {"digit1", "tmp"}:
            if variant_spec.add_extra_tmp2 and preds == {"digit1", "tmp2"}:
                pass
            else:
                return False

        (head_x,) = c.head.args  # sum_is(X)
        digit_atom = b1 if b1.pred == "digit1" else b2
        tmp_atom   = b2 if digit_atom is b1 else b1

        (z0,) = digit_atom.args
        if not isinstance(z0, Var) or z0.name != "Z0":
            return False

        if variant_spec.sum_filter_kind == "canonical_only":
            return tmp_atom.args == (head_x, z0)

        if MODE == "tight":
            allowed_tmp_args = {
                (head_x, z0),  # intended: tmp(S,A) with A from digit1
                (z0, head_x),  # "wrong" wiring
            }
            return tmp_atom.args in allowed_tmp_args

        allowed_tmp_args = {
            (head_x, z0),
            (z0, head_x),
            (head_x, head_x),
            (z0, z0)
        }
        return tmp_atom.args in allowed_tmp_args

    bias = BiasConfig(
        allowed_body_preds=variant_spec.allowed_body_preds,
        require_recursive={},
        require_body_connected=True,
        custom_clause_filters={
            ("tmp", 2):    tmp_mode_filter,
            ("sum_is", 1): sum_mode_filter,
            **({("tmp2", 2): tmp2_mode_filter} if variant_spec.add_extra_tmp2 else {}),
        },
    )

    aux_keys = [("tmp", 2)]
    if variant_spec.add_extra_tmp2:
        aux_keys.append(("tmp2", 2))

    return TaskConfig(
        constants=C,
        predicates=predicates,
        arg_domains=arg_domains,
        target_key=("sum_is", 1),
        aux_keys=aux_keys,
        templates=templates,
        T=T,
        bias=bias,
        require_recursive_on_C2=variant_spec.require_recursive_on_C2,
    )
