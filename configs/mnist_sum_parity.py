from __future__ import annotations

from dataclasses import dataclass

from ilp.learning.bias import BiasConfig
from ilp.learning.data import PredicateKey
from ilp.learning.task_config import TaskConfig
from ilp.logic.atoms import Predicate
from ilp.logic.clauses import Clause, Var
from ilp.logic.templates import RuleTemplate


@dataclass(frozen=True)
class _SumParityVariantSpec:
    name: str
    allowed_body_preds: dict[PredicateKey, set[str]]
    tmp_filter_kind: str
    sum_filter_kind: str
    parity_filter_kind: str


def _make_sum_parity_variant_spec(variant: str) -> _SumParityVariantSpec:
    if variant == "base":
        return _SumParityVariantSpec(
            name="base",
            allowed_body_preds={
                ("tmp", 2): {"digit2", "add"},
                ("sum_is", 1): {"digit1", "tmp"},
                ("sum_parity", 1): {"sum_is", "parity_of"},
            },
            tmp_filter_kind="mode",
            sum_filter_kind="mode",
            parity_filter_kind="mode",
        )

    if variant == "tmp_broad_only":
        return _SumParityVariantSpec(
            name="tmp_broad_only",
            allowed_body_preds={
                ("tmp", 2): {"digit1", "digit2", "add"},
                ("sum_is", 1): {"digit1", "tmp"},
                ("sum_parity", 1): {"sum_is", "parity_of"},
            },
            tmp_filter_kind="broad_structured",
            sum_filter_kind="mode",
            parity_filter_kind="mode",
        )

    if variant == "broad_search":
        return _SumParityVariantSpec(
            name="broad_search",
            allowed_body_preds={
                ("tmp", 2): {"digit1", "digit2", "add"},
                ("sum_is", 1): {"digit1", "digit2", "tmp", "add"},
                ("sum_parity", 1): {"tmp", "sum_is", "parity_of"},
            },
            tmp_filter_kind="broad_structured",
            sum_filter_kind="broad_structured",
            parity_filter_kind="broad_structured",
        )

    raise ValueError(
        "Unsupported MNIST-SumParity variant: "
        f"{variant}. Known variants: base, tmp_broad_only, broad_search"
    )


def make_config(
    mode: str = "tight",
    T: int = 3,
    variant: str = "base",
    n_digits: int = 10,
) -> TaskConfig:
    """
    ILP config for the biased MNIST-SumParity task.

    The dataset keeps only the in-distribution parity combinations
    (even, even), (odd, odd), and (odd, even), while (even, odd) pairs are
    held out for OOD evaluation. The target is the parity of the sum.
    """
    if mode not in {"tight", "medium"}:
        raise ValueError(f"Unsupported MNIST-SumParity mode: {mode}")
    if T <= 0:
        raise ValueError("T must be > 0")
    if n_digits <= 1:
        raise ValueError("n_digits must be > 1")

    mode_name = mode
    variant_spec = _make_sum_parity_variant_spec(variant)

    digits = [str(d) for d in range(n_digits)]
    sums = [str(s) for s in range((2 * n_digits) - 1)]
    parities = ["even", "odd"]
    constants = list(dict.fromkeys(digits + sums + parities))

    arg_domains = {
        ("digit1", 1): [digits],
        ("digit2", 1): [digits],
        ("sum_is", 1): [sums],
        ("tmp", 2): [sums, digits],
        ("add", 3): [digits, digits, sums],
        ("parity_of", 2): [sums, parities],
        ("sum_parity", 1): [parities],
    }

    predicates = [
        Predicate("digit1", 1, "E"),
        Predicate("digit2", 1, "E"),
        Predicate("add", 3, "E"),
        Predicate("parity_of", 2, "E"),
        Predicate("tmp", 2, "I"),
        Predicate("sum_is", 1, "I"),
        Predicate("sum_parity", 1, "I"),
    ]

    tau_tmp = RuleTemplate(v=1, int_flag=0)
    tau_sum = RuleTemplate(v=1, int_flag=1)
    tau_parity = RuleTemplate(v=1, int_flag=1)

    templates = {
        ("tmp", 2): (tau_tmp, tau_tmp),
        ("sum_is", 1): (tau_sum, tau_sum),
        ("sum_parity", 1): (tau_parity, tau_parity),
    }

    def tmp_mode_filter(c: Clause) -> bool:
        b1, b2 = c.body
        if variant_spec.tmp_filter_kind == "broad_structured":
            preds = {b1.pred, b2.pred}
            if "add" not in preds:
                return False

            digit_preds = preds.intersection({"digit1", "digit2"})
            if len(digit_preds) != 1:
                return False

            head_sum, head_digit = c.head.args
            digit_atom = b1 if b1.pred in {"digit1", "digit2"} else b2
            add_atom = b2 if digit_atom is b1 else b1

            (digit_var,) = digit_atom.args
            if not isinstance(digit_var, Var):
                return False

            add_args = set(add_atom.args)
            return head_sum in add_args and head_digit in add_args and digit_var in add_args

        preds = {b1.pred, b2.pred}
        if preds != {"digit2", "add"}:
            return False

        head_sum, head_digit = c.head.args
        digit_atom = b1 if b1.pred == "digit2" else b2
        add_atom = b2 if digit_atom is b1 else b1

        (z0,) = digit_atom.args
        if not isinstance(z0, Var) or z0.name != "Z0":
            return False

        if mode_name == "tight":
            allowed_add_args = {
                (head_digit, z0, head_sum),
                (z0, head_digit, head_sum),
            }
            return add_atom.args in allowed_add_args

        allowed_add_args = {
            (head_digit, z0, head_sum),
            (z0, head_digit, head_sum),
            (head_sum, head_digit, z0),
            (head_sum, z0, head_digit),
            (head_digit, head_sum, z0),
            (z0, head_sum, head_digit),
        }
        return add_atom.args in allowed_add_args

    def sum_mode_filter(c: Clause) -> bool:
        b1, b2 = c.body
        if variant_spec.sum_filter_kind == "broad_structured":
            preds = {b1.pred, b2.pred}
            allowed_pred_pairs = {
                frozenset({"digit1", "tmp"}),
                frozenset({"digit2", "tmp"}),
                frozenset({"digit1", "add"}),
                frozenset({"digit2", "add"}),
            }
            if frozenset(preds) not in allowed_pred_pairs:
                return False

            (head_sum,) = c.head.args
            digit_atom = b1 if b1.pred in {"digit1", "digit2"} else b2
            rel_atom = b2 if digit_atom is b1 else b1

            (digit_var,) = digit_atom.args
            if not isinstance(digit_var, Var):
                return False

            rel_args = set(rel_atom.args)
            return head_sum in rel_args and digit_var in rel_args

        preds = {b1.pred, b2.pred}
        if preds != {"digit1", "tmp"}:
            return False

        (head_sum,) = c.head.args
        digit_atom = b1 if b1.pred == "digit1" else b2
        tmp_atom = b2 if digit_atom is b1 else b1

        (z0,) = digit_atom.args
        if not isinstance(z0, Var) or z0.name != "Z0":
            return False

        return tmp_atom.args == (head_sum, z0)

    def parity_mode_filter(c: Clause) -> bool:
        b1, b2 = c.body
        if variant_spec.parity_filter_kind == "broad_structured":
            preds = {b1.pred, b2.pred}
            allowed_pred_pairs = {
                frozenset({"sum_is", "parity_of"}),
                frozenset({"tmp", "parity_of"}),
            }
            if frozenset(preds) not in allowed_pred_pairs:
                return False

            (head_parity,) = c.head.args
            rel_atom = b1 if b1.pred != "parity_of" else b2
            parity_atom = b2 if rel_atom is b1 else b1

            if head_parity not in parity_atom.args:
                return False

            shared_vars = {
                arg
                for arg in rel_atom.args
                if isinstance(arg, Var) and arg in parity_atom.args
            }
            return len(shared_vars) > 0

        preds = {b1.pred, b2.pred}
        if preds != {"sum_is", "parity_of"}:
            return False

        (head_parity,) = c.head.args
        sum_atom = b1 if b1.pred == "sum_is" else b2
        parity_atom = b2 if sum_atom is b1 else b1

        (z0,) = sum_atom.args
        if not isinstance(z0, Var) or z0.name != "Z0":
            return False

        return parity_atom.args == (z0, head_parity)

    bias = BiasConfig(
        allowed_body_preds=variant_spec.allowed_body_preds,
        require_recursive={},
        require_body_connected=True,
        custom_clause_filters={
            ("tmp", 2): tmp_mode_filter,
            ("sum_is", 1): sum_mode_filter,
            ("sum_parity", 1): parity_mode_filter,
        },
    )

    return TaskConfig(
        constants=constants,
        predicates=predicates,
        arg_domains=arg_domains,
        target_key=("sum_parity", 1),
        aux_keys=[("tmp", 2), ("sum_is", 1)],
        templates=templates,
        T=T,
        bias=bias,
        require_recursive_on_C2={
            ("tmp", 2): False,
            ("sum_is", 1): False,
            ("sum_parity", 1): False,
        },
    )
