from __future__ import annotations

from ilp.logic.atoms import Predicate
from ilp.logic.clauses import Clause, Var
from ilp.logic.templates import RuleTemplate
from ilp.learning.bias import BiasConfig
from ilp.learning.task_config import TaskConfig


def make_config(mode: str = "medium", T: int = 2, variant: str = "base") -> TaskConfig:
    """
    ILP config for the RSBench MNIST-Even-Odd dataset.

    The dataset is implemented in `third_party/rsbench-code/rsseval/rss/datasets/shortcutmnist.py`
    as `SHORTMNIST`, with the in-distribution digit pairs:
        (0,6), (2,8), (4,6), (4,8), (1,5), (3,7), (1,9), (3,9)
    plus their swapped versions. The observed in-distribution sums are {6, 10, 12}.

    We keep the symbolic `add/3` relation fully grounded over digits 0..9 and sums 0..18,
    so the learner can still assign mass to unseen sums at evaluation time.
    """
    if mode not in {"tight", "medium"}:
        raise ValueError(f"Unsupported MNIST-Even-Odd addition mode: {mode}")
    if T <= 0:
        raise ValueError("T must be > 0")
    if variant != "base":
        raise ValueError(
            "Unsupported MNIST-Even-Odd addition variant: "
            f"{variant}. Known variants: base"
        )

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

    predicates = [
        Predicate("digit1", 1, "E"),
        Predicate("digit2", 1, "E"),
        Predicate("add", 3, "E"),
        Predicate("tmp", 2, "I"),
        Predicate("sum_is", 1, "I"),
    ]

    tau_tmp = RuleTemplate(v=1, int_flag=0)
    tau_sum = RuleTemplate(v=1, int_flag=1)
    templates = {
        ("tmp", 2): (tau_tmp, tau_tmp),
        ("sum_is", 1): (tau_sum, tau_sum),
    }

    def tmp_mode_filter(c: Clause) -> bool:
        b1, b2 = c.body
        preds = {b1.pred, b2.pred}
        if preds != {"digit2", "add"}:
            return False

        head_x, head_y = c.head.args
        digit_atom = b1 if b1.pred == "digit2" else b2
        add_atom = b2 if digit_atom is b1 else b1

        (z0,) = digit_atom.args
        if not isinstance(z0, Var) or z0.name != "Z0":
            return False

        if mode == "tight":
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

        if mode == "tight":
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

    return TaskConfig(
        constants=constants,
        predicates=predicates,
        arg_domains=arg_domains,
        target_key=("sum_is", 1),
        aux_keys=[("tmp", 2)],
        templates=templates,
        T=T,
        bias=bias,
        require_recursive_on_C2={
            ("tmp", 2): False,
            ("sum_is", 1): False,
        },
    )
