from __future__ import annotations

from dataclasses import dataclass

from ilp.learning.bias import BiasConfig
from ilp.learning.data import PredicateKey
from ilp.learning.task_config import TaskConfig
from ilp.logic.atoms import Predicate
from ilp.logic.clauses import Clause, Var
from ilp.logic.templates import RuleTemplate


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
        "Unsupported MNIST-Even-Odd addition variant: "
        f"{variant}. Known variants: base, broad_search, sum_relaxed, extra_tmp2"
    )


def make_config(mode: str = "medium", T: int = 2, variant: str = "base") -> TaskConfig:
    """
    ILP config for the RSBench MNIST-Even-Odd dataset.

    The dataset is implemented in `third_party/rsbench-code/rsseval/rss/datasets/shortcutmnist.py`
    as `SHORTMNIST`, with the in-distribution digit pairs:
        (0,6), (2,8), (4,6), (4,8), (1,5), (3,7), (1,9), (3,9)
    plus their swapped versions. The observed in-distribution sums are {6, 10, 12}.

    The symbolic add/3 relation is fully grounded over digits 0..9 and sums 0..18.
    """
    if mode not in {"tight", "medium"}:
        raise ValueError(f"Unsupported MNIST-Even-Odd addition mode: {mode}")
    if T <= 0:
        raise ValueError("T must be > 0")

    mode_name = mode
    variant_spec = _make_addition_variant_spec(variant)

    digits = [str(d) for d in range(10)]
    sums = [str(s) for s in range(19)]
    constants = list(dict.fromkeys(digits + sums))

    arg_domains = {
        ("digit1", 1): [digits],
        ("digit2", 1): [digits],
        ("sum_is", 1): [sums],
        ("tmp", 2): [sums, digits],
        ("add", 3): [digits, digits, sums],
    }
    if variant_spec.add_extra_tmp2:
        arg_domains[("tmp2", 2)] = [sums, digits]

    predicates = [
        Predicate("digit1", 1, "E"),
        Predicate("digit2", 1, "E"),
        Predicate("add", 3, "E"),
        Predicate("tmp", 2, "I"),
        Predicate("sum_is", 1, "I"),
    ]
    if variant_spec.add_extra_tmp2:
        predicates.insert(4, Predicate("tmp2", 2, "I"))

    tau_tmp = RuleTemplate(v=1, int_flag=0)
    tau_sum = RuleTemplate(v=1, int_flag=1)
    templates = {
        ("tmp", 2): (tau_tmp, tau_tmp),
        ("sum_is", 1): (tau_sum, tau_sum),
    }
    if variant_spec.add_extra_tmp2:
        templates[("tmp2", 2)] = (tau_tmp, tau_tmp)

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

        head_x, head_y = c.head.args
        digit_atom = b1 if b1.pred in {"digit1", "digit2"} else b2
        add_atom = b2 if digit_atom is b1 else b1

        (digit_var,) = digit_atom.args
        if not isinstance(digit_var, Var):
            return False

        if filter_kind == "broad_structured":
            return digit_var in add_atom.args

        z0 = digit_var
        if z0.name != "Z0":
            return False

        if mode_name == "tight":
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
            (z0, head_x, head_y),
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

        (head_x,) = c.head.args
        digit_atom = b1 if b1.pred == "digit1" else b2
        tmp_atom = b2 if digit_atom is b1 else b1

        (z0,) = digit_atom.args
        if not isinstance(z0, Var) or z0.name != "Z0":
            return False

        if mode_name == "tight":
            allowed_tmp_args = {
                (head_x, z0),
                (z0, head_x),
            }
            return tmp_atom.args in allowed_tmp_args

        allowed_tmp_args = {
            (head_x, z0),
            (z0, head_x),
            (head_x, head_x),
            (z0, z0),
        }
        return tmp_atom.args in allowed_tmp_args

    bias = BiasConfig(
        allowed_body_preds=variant_spec.allowed_body_preds,
        require_recursive={},
        require_body_connected=True,
        custom_clause_filters={
            ("tmp", 2): tmp_mode_filter,
            ("sum_is", 1): sum_mode_filter,
            **({("tmp2", 2): tmp2_mode_filter} if variant_spec.add_extra_tmp2 else {}),
        },
    )

    aux_keys = [("tmp", 2)]
    if variant_spec.add_extra_tmp2:
        aux_keys.append(("tmp2", 2))

    return TaskConfig(
        constants=constants,
        predicates=predicates,
        arg_domains=arg_domains,
        target_key=("sum_is", 1),
        aux_keys=aux_keys,
        templates=templates,
        T=T,
        bias=bias,
        require_recursive_on_C2=variant_spec.require_recursive_on_C2,
    )
