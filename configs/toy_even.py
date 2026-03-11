from __future__ import annotations

from ilp.logic.atoms import Predicate
from ilp.logic.templates import RuleTemplate
from ilp.learning.bias import BiasConfig
from ilp.learning.task_config import TaskConfig


def make_config() -> TaskConfig:
    C = ["0", "1", "2", "3", "4", "5"]

    predicates = [
        Predicate("zero", 1, "E"),
        Predicate("succ", 2, "E"),
        Predicate("even", 1, "I"),
        Predicate("succ2", 2, "I"),
    ]

    templates = {
        ("even", 1): (RuleTemplate(v=0, int_flag=0), RuleTemplate(v=1, int_flag=1)),
        ("succ2", 2): (RuleTemplate(v=1, int_flag=0), RuleTemplate(v=1, int_flag=0)),
    }

    bias = BiasConfig(
        allowed_body_preds={
            ("succ2", 2): {"succ"},
            ("even", 1): {"zero", "succ2", "even"},
        },
        require_recursive={},
        require_body_connected=True,
    )

    require_recursive_on_C2 = {
        ("even", 1): True
    }

    return TaskConfig(
        constants=C,
        predicates=predicates,
        arg_domains={},
        target_key=("even", 1),
        aux_keys=[("succ2", 2)],
        templates=templates,
        T=4,
        bias=bias,
        require_recursive_on_C2=require_recursive_on_C2,
    )