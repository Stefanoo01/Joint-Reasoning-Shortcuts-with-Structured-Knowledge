from __future__ import annotations

from logic.atoms import Predicate
from logic.templates import RuleTemplate
from learning.bias import BiasConfig
from learning.task_config import TaskConfig
from logic.clauses import Clause, Var

def make_config() -> TaskConfig:
    """
    ILP config for HALFMNIST "Odd Sum" detection.
    
    Instead of passing the mathematical `add(A,B,S)` table, we ask the model
    to predict a binary target: `is_odd()`.
    
    In HALFMNIST, the valid sums are 0, 1, 5, 6.
    The ODD sums (1 and 5) are formed EXCLUSIVELY by pairs: (0,1), (1,0), (2,3), (3,2).
    Notice that in EVERY odd pair, one digit is the exact successor of the other!
    The EVEN sums (0 and 6) are formed by (0,0), (2,4), (4,2). None are successors.
    
    Therefore, the ILP can perfectly classify `is_odd` by discovering:
        is_odd() :- digit1(A), digit2(B), succ(A, B)
        is_odd() :- digit1(A), digit2(B), succ(B, A)
        
    We provide `succ/2` as base knowledge, along with distractor rules
    like `zero/1` or `even/1` to encourage wild reasoning shortcuts!
    """

    # Domain
    digits = [str(d) for d in range(5)]

    # Typed grounding domains
    arg_domains = {
        ("digit1", 1): [digits],
        ("digit2", 1): [digits],
        ("is_odd", 0): [],
        ("tmp", 2):    [digits, digits], 
        ("succ", 2):   [digits, digits],
        ("zero", 1):   [digits],
        ("even_num", 1): [digits],
    }

    predicates = [
        Predicate("digit1", 1, "E"),
        Predicate("digit2", 1, "E"),
        Predicate("succ",   2, "E"),   # Extensional!
        Predicate("zero",   1, "E"),   # Distractor
        Predicate("even_num", 1, "E"), # Distractor
        Predicate("tmp",    2, "I"),   # Auxiliary relation between A and B
        Predicate("is_odd", 0, "I"),   # Target
    ]

    # Templates
    # tmp(A,B) needs 0 existential vars, it operates exactly on head vars
    tau_tmp = RuleTemplate(v=0, int_flag=0)
    
    # is_odd() needs 2 existential vars (the two digits)
    tau_odd = RuleTemplate(v=2, int_flag=1)

    templates = {
        ("tmp", 2):    (tau_tmp, tau_tmp),
        ("is_odd", 0): (tau_odd, tau_odd), # We allow two rules for OR logic
    }

    def tmp_mode_filter(c: Clause) -> bool:
        """
        Force tmp(X,Y) to use succ. But because we have distractors, we allow it
        to combine with zero or even_num to see if it takes shortcuts.
        """
        # Very light filter: just make sure it's not circular
        return not c.is_circular()

    def is_odd_filter(c: Clause) -> bool:
        """
        is_odd() :- digit1(Z0), digit2(Z1), tmp(Z0,Z1)
        We let the network explore connections.
        """
        b1, b2 = c.body
        preds = {b1.pred, b2.pred}
        # Force it to wire up digit1 and digit2 to the learned tmp relation
        if "digit1" in preds or "digit2" in preds:
            return True
        return False

    bias = BiasConfig(
        allowed_body_preds={
            ("tmp", 2):    {"succ", "zero", "even_num"},
            ("is_odd", 0): {"digit1", "digit2", "tmp"},
        },
        require_recursive={},
        require_body_connected=True,
        custom_clause_filters={
            ("tmp", 2):    tmp_mode_filter,
            ("is_odd", 0): is_odd_filter,
        },
    )

    return TaskConfig(
        constants=digits,
        predicates=predicates,
        arg_domains=arg_domains,
        target_key=("is_odd", 0),
        aux_keys=[("tmp", 2)],
        templates=templates,
        T=2,
        bias=bias,
        require_recursive_on_C2={("is_odd", 0): False, ("tmp", 2): False},
    )
