from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from logic.atoms import Atom, Predicate
from logic.language import LanguageSpec, build_ground_atoms, build_index
from logic.templates import ProgramTemplate

from learning.task_config import TaskConfig
from learning.data import PredicateKey
from learning.build_program import build_clause_sets_for_program, build_caches_with_bias
from learning.model import ProgramLearner


@dataclass
class SystemBundle:
    spec: LanguageSpec
    G: list[Atom]
    atom_to_idx: Dict[Atom, int]
    idx_to_atom: list[Atom]
    bot_idx: int
    learner: ProgramLearner
    clause_texts: dict[PredicateKey, tuple[list[str], list[str]]]
    program: ProgramTemplate
    target_predicate: Predicate
    aux_predicates: list[Predicate]


def build_system_from_config(cfg: TaskConfig) -> SystemBundle:
    # 1) Language
    spec = LanguageSpec(constants=cfg.constants, predicates=cfg.predicates)
    G = build_ground_atoms(spec)
    atom_to_idx, idx_to_atom, bot_idx = build_index(G)
    print()

    # 2) Build predicates objects for target/aux from cfg.predicates
    def find_pred(key: PredicateKey) -> Predicate:
        name, arity = key
        return next(p for p in cfg.predicates if p.name == name and p.arity == arity)

    target_pred = find_pred(cfg.target_key)
    aux_preds = [find_pred(k) for k in cfg.aux_keys]

    # 3) Π
    Pi = ProgramTemplate(
        aux_predicates=aux_preds,
        rules={k: cfg.templates[k] for k in cfg.templates.keys()},
        T=cfg.T,
    )

    # 4) Clause sets -> caches (with bias)
    clause_sets = build_clause_sets_for_program(
        predicates=cfg.predicates,
        target_pred=target_pred,
        program=Pi,
    )

    caches, clause_texts = build_caches_with_bias(
        clause_sets=clause_sets,
        constants=cfg.constants,
        atom_to_idx=atom_to_idx,
        n=len(G),
        bot_idx=bot_idx,
        bias=cfg.bias,
        require_recursive_on_C2=cfg.require_recursive_on_C2,
    )

    learner = ProgramLearner(caches)

    return SystemBundle(
        spec=spec,
        G=G,
        atom_to_idx=atom_to_idx,
        idx_to_atom=idx_to_atom,
        bot_idx=bot_idx,
        learner=learner,
        clause_texts=clause_texts,
        program=Pi,
        target_predicate=target_pred,
        aux_predicates=aux_preds,
    )