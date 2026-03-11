from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ilp.logic.atoms import Predicate
from ilp.logic.templates import RuleTemplate
from ilp.learning.data import PredicateKey
from ilp.learning.bias import BiasConfig


@dataclass(frozen=True)
class TaskConfig:
    # Language
    constants: List[str]
    predicates: List[Predicate]
    arg_domains: Dict[PredicateKey, List[List[str]]]

    # Target / aux
    target_key: PredicateKey
    aux_keys: List[PredicateKey]

    # Π (templates + inference depth)
    templates: Dict[PredicateKey, Tuple[RuleTemplate, RuleTemplate]]
    T: int

    # Bias
    bias: BiasConfig
    require_recursive_on_C2: Dict[PredicateKey, bool]