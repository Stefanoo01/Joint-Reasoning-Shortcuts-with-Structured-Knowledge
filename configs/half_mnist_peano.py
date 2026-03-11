from __future__ import annotations

from ilp.logic.atoms import Predicate
from ilp.logic.templates import RuleTemplate
from ilp.learning.bias import BiasConfig
from ilp.learning.task_config import TaskConfig
from ilp.logic.clauses import Clause, Var

def make_config() -> TaskConfig:
    """
    ILP config for HALFMNIST Addidtion using Peano Axioms (Recursive).
    """
    
    digits = [str(d) for d in range(5)]
    sums   = [str(s) for s in range(9)]
    
    raw = digits + sums
    C = list(dict.fromkeys(raw))

    arg_domains = {
        ("digit1", 1): [digits],
        ("digit2", 1): [digits],
        ("succ",   2): [sums, sums],
        ("zero",   1): [sums],
        ("step",   3): [sums, sums, digits], # step(S, PrevS, A): "S is PrevS + 1, and A could have triggered it" 
        ("tmp",    2): [digits, sums],       # tmp(A, S) meaning "digit2 + A = S"
        ("sum_is", 1): [sums],
    }

    predicates = [
        Predicate("digit1", 1, "E"),
        Predicate("digit2", 1, "E"),
        Predicate("succ",   2, "E"),   
        Predicate("zero",   1, "E"),   
        Predicate("step",   3, "E"),   
        Predicate("tmp",    2, "I"),   
        Predicate("sum_is", 1, "I"),   
    ]

    # RULES WE WANT TO ALLOW:
    # 1. Base case
    # tmp(A, S) :- zero(A), digit2(S)
    
    # 2. Recursive case
    # tmp(A, S) :- step(S, PrevS, A), tmp(PrevA_EXISTENTIAL, PrevS)
    # The trick: `step(S, PrevS, A)` is ONLY TRUE when S=PrevS+1 AND PrevA_EXISTENTIAL = A-1.
    # We will build `step` in python so that it perfectly aligns PrevS and A, acting as a double-variable binder.
    tau_tmp_base = RuleTemplate(v=0, int_flag=0)
    tau_tmp_rec  = RuleTemplate(v=2, int_flag=1) # 2 existential vars: Z0, Z1 (we'll map Z1=PrevS, Z0=PrevA)
    
    tau_sum = RuleTemplate(v=1, int_flag=1)

    templates = {
        ("tmp", 2):    (tau_tmp_base, tau_tmp_rec),
        ("sum_is", 1): (tau_sum, tau_sum),
    }

    def tmp_filter(c: Clause) -> bool:
        b1, b2 = c.body
        preds = {b1.pred, b2.pred}
        
        # Base case
        if "zero" in preds:
            return True
        # Recursive case
        if "step" in preds:
            return True
            
        return False

    def sum_filter(c: Clause) -> bool:
        b1, b2 = c.body
        preds = {b1.pred, b2.pred}
        if preds == {"digit1", "tmp"}:
            return True
        return False

    bias = BiasConfig(
        allowed_body_preds={
            ("tmp", 2):    {"zero", "digit2", "step", "tmp"},
            ("sum_is", 1): {"digit1", "tmp"},
        },
        require_recursive={},
        require_body_connected=True,
        custom_clause_filters={
            ("tmp", 2):    tmp_filter,
            ("sum_is", 1): sum_filter,
        },
    )

    return TaskConfig(
        constants=C,
        predicates=predicates,
        arg_domains=arg_domains,
        target_key=("sum_is", 1),
        aux_keys=[("tmp", 2)],
        templates=templates,
        T=5, 
        bias=bias,
        require_recursive_on_C2={
            ("tmp", 2): True,
            ("sum_is", 1): False,
        },
    )
