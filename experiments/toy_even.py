from __future__ import annotations
import torch
from logic.atoms import Atom, BOT, Predicate
from logic.language import LanguageSpec, build_ground_atoms, build_index
from logic.valuation import build_a0
from logic.templates import RuleTemplate, ProgramTemplate
from logic.clauses import generate_clauses_for_template
from compile.compile_clause import compile_clause_to_X
from logic.infer import f_clause, infer_one_step, infer_T
from learning.data import build_targets_from_positives
from learning.compile_cache import compile_clause_set_stack
from learning.model import ProgramLearner, PredicateClauseCache
from learning.trainer import train_program, TrainConfig
from learning.bias import BiasConfig
from learning.build_program import build_clause_sets_for_program, build_caches_with_bias
from logic.valuation_soft import build_a0_from_facts

def main() -> None:
    # Constants
    C = ["0", "1", "2", "3", "4", "5"]

    # Predicates: extensional + intensional
    predicates = [
        Predicate("zero", 1, "E"),
        Predicate("succ", 2, "E"),
        Predicate("even", 1, "I"),
        Predicate("succ2", 2, "I"),   # NEW aux intensional
    ]

    target_pred = next(p for p in predicates if p.name == "even" and p.arity == 1)
    succ2_pred  = next(p for p in predicates if p.name == "succ2" and p.arity == 2)

    # Build language and ground atoms
    spec = LanguageSpec(constants=C, predicates=predicates)

    G = build_ground_atoms(spec)
    atom_to_idx, idx_to_atom, bot_idx = build_index(G)

    # Assertions (tests)
    expected_n = 6 + 36 * 2 + 6 + 1
    assert len(G) == expected_n, f"Expected |G|={expected_n}, got {len(G)}"

    assert G[-1] == BOT, "Expected BOT to be the last atom in G"
    assert idx_to_atom[bot_idx] == BOT, "bot_idx does not map to BOT"
    assert BOT in atom_to_idx, "BOT missing from atom_to_idx"

    # Check some specific atoms exist
    assert Atom("zero", ("0",)) in atom_to_idx
    assert Atom("succ", ("2", "3")) in atom_to_idx
    assert Atom("even", ("4",)) in atom_to_idx

    # Bijectivity sanity check
    for a, i in atom_to_idx.items():
        assert idx_to_atom[i] == a

    print(f"|G| = {len(G)}  (expected {expected_n})")
    print(f"bot_idx = {bot_idx}, BOT = {idx_to_atom[bot_idx]}")

    B = [
        Atom("zero", ("0",)),
        Atom("succ", ("0", "1")),
        Atom("succ", ("1", "2")),
        Atom("succ", ("2", "3")),
        Atom("succ", ("3", "4")),
        Atom("succ", ("4", "5")),
    ]

    n = len(G)
    a0 = build_a0(n=n, atom_to_idx=atom_to_idx, B=B, bot_idx=bot_idx)

    # Assertions
    assert len(a0) == n
    assert a0[atom_to_idx[Atom("zero", ("0",))]] == 1.0
    assert a0[atom_to_idx[Atom("succ", ("2", "3"))]] == 1.0
    assert a0[atom_to_idx[Atom("zero", ("1",))]] == 0.0
    assert a0[atom_to_idx[Atom("even", ("0",))]] == 0.0
    assert a0[atom_to_idx[Atom("even", ("2",))]] == 0.0
    assert a0[bot_idx] == 0.0

    print(f"a0 ones = {sum(1 for x in a0 if x == 1.0)} (expected 6)")

    a0_soft = build_a0_from_facts(
        n=n,
        atom_to_idx=atom_to_idx,
        bot_idx=bot_idx,
        hard_facts=B,
        soft_facts=[],
    )

    assert float(a0_soft[bot_idx].item()) == 0.0
    assert float(a0_soft[atom_to_idx[Atom("zero", ("0",))]].item()) == 1.0
    assert float(a0_soft[atom_to_idx[Atom("succ", ("2", "3"))]].item()) == 1.0
    assert float(a0_soft[atom_to_idx[Atom("zero", ("1",))]].item()) == 0.0

    # test soft fact
    idx_even2 = atom_to_idx[Atom("even", ("2",))]
    a0_soft2 = build_a0_from_facts(
        n=n,
        atom_to_idx=atom_to_idx,
        bot_idx=bot_idx,
        hard_facts=B,
        soft_facts=[(Atom("even", ("2",)), 0.7)],
    )
    assert abs(float(a0_soft2[idx_even2].item()) - 0.7) < 1e-6

    print("OK: valuation_soft.build_a0_from_facts mini-tests passed")

    tau1_even = RuleTemplate(v=0, int_flag=0)
    tau2_even = RuleTemplate(v=1, int_flag=1)
    tau_succ2 = RuleTemplate(v=1, int_flag=0)

    rules = {
        ("even", 1): (tau1_even, tau2_even),
        ("succ2", 2): (tau_succ2, tau_succ2),
    }

    Pi = ProgramTemplate(
        aux_predicates=[succ2_pred],
        rules=rules,
        T=4,
    )

    # Assertions
    assert Pi.T == 4
    assert ("even", 1) in Pi.rules
    assert Pi.rules[("even", 1)][0] == tau1_even
    assert Pi.rules[("even", 1)][1] == tau2_even

    # intensional predicates should include target
    Pi_int = Pi.intensional_predicates(target_pred)
    assert any(p.name == "even" and p.arity == 1 for p in Pi_int)
    assert len(Pi.aux_predicates) == 1

    print(f"Π: T={Pi.T}, aux={len(Pi.aux_predicates)}, rules_keys={list(Pi.rules.keys())}")

    tau1, tau2 = Pi.rules[("even", 1)]
    C1 = generate_clauses_for_template(head_pred=target_pred, predicates=predicates, template=tau1)
    C2 = generate_clauses_for_template(head_pred=target_pred, predicates=predicates, template=tau2)

    tau1_succ2, tau2_succ2 = Pi.rules[("succ2", 2)]
    S1 = generate_clauses_for_template(head_pred=succ2_pred, predicates=predicates, template=tau1_succ2)
    S2 = generate_clauses_for_template(head_pred=succ2_pred, predicates=predicates, template=tau2_succ2)

    succ2_clause = next(
        c for c in S1
        if str(c) == "succ2(X,Y) :- succ(X,Z0), succ(Z0,Y)."
    )

    print("succ2_clause:", succ2_clause)

    # Assertions
    assert len(C1) > 0, "No clauses generated for tau1"
    assert len(C2) > 0, "No clauses generated for tau2"

    # All clauses must be safe and non-circular
    assert all(c.is_safe() for c in C1)
    assert all(c.is_safe() for c in C2)
    assert all(not c.is_circular() for c in C1)
    assert all(not c.is_circular() for c in C2)

    # Duplicate under swap should be eliminated
    assert len({c.canonical_key() for c in C1}) == len(C1)
    assert len({c.canonical_key() for c in C2}) == len(C2)

    assert any(str(c) == "even(X) :- zero(X), zero(X)." for c in C1), "Missing base clause in C1"
    assert any(str(c) == "even(X) :- zero(X), zero(X)." for c in C2), "Missing base clause in C2"
    assert all(("even(" not in str(c.body[0]) and "even(" not in str(c.body[1])) for c in C1), "C1 should not contain intensional predicates"

    assert all(str(c) != "even(X) :- zero(X), even(Z0)." for c in C2)

    print(f"|C1| (tau1) = {len(C1)}")
    print(f"|C2| (tau2) = {len(C2)}")
    print("Example C1:", C1[0])
    print("Example C2:", C2[0])
    rec = [c for c in C2 if ("even(" in str(c.body[0]) or "even(" in str(c.body[1]))]
    print("recursive C2:", len(rec))
    print("example recursive:", rec[0] if rec else None)

    base_clause = next(c for c in C1 if str(c) == "even(X) :- zero(X), zero(X).")
    rec_clause = next(
        c for c in C2
        if ("even(" in str(c.body[0]) or "even(" in str(c.body[1]))
    )

    even_rec_clause = None
    for c in C2:
        # vogliamo head: even(X)
        if c.head.pred != "even":
            continue
        (X,) = c.head.args

        b1, b2 = c.body
        preds = {b1.pred, b2.pred}
        if preds != {"succ2", "even"}:
            continue

        succ_atom = b1 if b1.pred == "succ2" else b2
        even_atom = b2 if succ_atom is b1 else b1

        # succ2(Z0, X) e even(Z0)
        if succ_atom.args[1] == X and even_atom.args[0] == succ_atom.args[0]:
            even_rec_clause = c
            break

    assert even_rec_clause is not None, "No suitable even recursion clause found"
    print("even_rec_clause:", even_rec_clause)

    constants = C  # reuse your constants list
    n = len(G)

    X_base = compile_clause_to_X(
        clause=base_clause,
        constants=constants,
        atom_to_idx=atom_to_idx,
        n=n,
        bot_idx=bot_idx,
    )
    Xb = torch.tensor(X_base, dtype=torch.long)
    
    assert Xb.shape[0] == n and Xb.shape[2] == 2
    assert Xb.dtype == torch.long

    # For base_clause, there are no existential vars -> w = 1
    assert Xb.shape[1] == 1

    X_rec = compile_clause_to_X(
        clause=rec_clause,
        constants=constants,
        atom_to_idx=atom_to_idx,
        n=n,
        bot_idx=bot_idx,
    )
    Xr = torch.tensor(X_rec, dtype=torch.long)
    assert Xr.shape[0] == n and Xr.shape[2] == 2
    assert Xr.dtype == torch.long

    for c in constants:
        k = atom_to_idx[Atom("even", (c,))]
        z = atom_to_idx[Atom("zero", (c,))]
        pair = tuple(Xb[k, 0].tolist())
        assert pair == (z, z), f"Base clause mismatch for even({c}): got {pair}, expected {(z,z)}"

    some_non_head = atom_to_idx[Atom("succ", ("0", "1"))]
    assert tuple(Xb[some_non_head, 0].tolist()) == (bot_idx, bot_idx)

    print("Base clause:", base_clause)
    print("Recursive clause example:", rec_clause)
    print("Xb shape:", tuple(Xb.shape), "Xr shape:", tuple(Xr.shape))

    X_succ2 = torch.tensor(
        compile_clause_to_X(succ2_clause, constants, atom_to_idx, n=len(G), bot_idx=bot_idx),
        dtype=torch.long
    )
    Xs2 = X_succ2  # [n, |C|, 2] perché Z0 esistenziale -> w=6

    X_even_base = Xb  # ce l'hai già
    X_even_rec = torch.tensor(
        compile_clause_to_X(even_rec_clause, constants, atom_to_idx, n=len(G), bot_idx=bot_idx),
        dtype=torch.long
    )

    print("Xs2 shape:", tuple(Xs2.shape))
    print("X_even_base shape:", tuple(X_even_base.shape))
    print("X_even_rec shape:", tuple(X_even_rec.shape))

    a = torch.tensor(a0, dtype=torch.float32)

    Fb = f_clause(a, Xb)  # [n]

    # For the base clause even(X) :- zero(X), zero(X):
    # even(c) should become 1 iff zero(c) is 1 in a0 (only c=0 in our B)
    for c in constants:
        k = atom_to_idx[Atom("even", (c,))]
        expected = 1.0 if c == "0" else 0.0
        got = float(Fb[k].item())
        assert abs(got - expected) < 1e-6, f"Fb mismatch for even({c}): got {got}, expected {expected}"

    print("idx even(0) =", atom_to_idx[Atom("even", ("0",))])
    print("Fb[even(0)] =", Fb[atom_to_idx[Atom("even", ("0",))]])

    a1 = infer_one_step(a, [Xb])

    # Facts from B must remain true (monotonicity via soft_or)
    assert float(a1[atom_to_idx[Atom("zero", ("0",))]].item()) == 1.0
    assert float(a1[atom_to_idx[Atom("succ", ("2", "3"))]].item()) == 1.0

    # even(0) should be 1 after the step, others 0 (because only base clause)
    for c in constants:
        idx = atom_to_idx[Atom("even", (c,))]
        expected = 1.0 if c == "0" else 0.0
        got = float(a1[idx].item())
        assert abs(got - expected) < 1e-6, f"a1 mismatch even({c}): got {got}, expected {expected}"

    """
    ones = (a1 == 1.0).nonzero(as_tuple=True)[0].tolist()
    print("Indices with 1:", ones)
    for i in ones:
        print(i, "->", idx_to_atom[i])
    """

    aT = infer_T(a, [Xb], T=Pi.T)

    # Con sola base clause, anche dopo T step deve rimanere solo even(0)=1
    for c in constants:
        idx = atom_to_idx[Atom("even", (c,))]
        expected = 1.0 if c == "0" else 0.0
        got = float(aT[idx].item())
        assert abs(got - expected) < 1e-6, f"aT mismatch even({c}): got {got}, expected {expected}"

    ones = (aT == 1.0).nonzero(as_tuple=True)[0].tolist()
    print("Indices with 1:", ones)
    for i in ones:
        print(i, "->", idx_to_atom[i])

    a = torch.tensor(a0, dtype=torch.float32)
    aT = infer_T(a, [Xs2, X_even_base, X_even_rec], T=Pi.T)

    for c in constants:
        print(f"even({c}) =", float(aT[atom_to_idx[Atom('even',(c,))]]))

    def filter_succ2_clauses(clauses):
        out = []
        for c in clauses:
            # body predicates must be exactly succ and succ
            if {b.pred for b in c.body} != {"succ"}:
                continue
            # must be connected (shared variable), altrimenti è roba tipo succ(X,X), succ(Y,Y)
            if not c.is_body_connected():
                continue
            out.append(c)
        return out

    S1_f = filter_succ2_clauses(S1)
    S2_f = filter_succ2_clauses(S2)

    print("succ2 filtered:", len(S1), "->", len(S1_f), " | ", len(S2), "->", len(S2_f))

    def filter_even_tau2_recursive(clauses):
        out = []
        for c in clauses:
            if not any(b.pred == "even" for b in c.body):
                continue
            if not c.is_body_connected():
                continue
            out.append(c)
        return out

    C2_f = filter_even_tau2_recursive(C2)
    print("even C2 filtered:", len(C2), "->", len(C2_f))

    clause_texts = {
        ("even", 1): ([str(c) for c in C1], [str(c) for c in C2_f]),
        ("succ2", 2): ([str(c) for c in S1_f], [str(c) for c in S2_f]),
    }

    caches = {
        ("even", 1): PredicateClauseCache(
            X1=compile_clause_set_stack(C1, C, atom_to_idx, n=len(G), bot_idx=bot_idx),
            X2=compile_clause_set_stack(C2_f, C, atom_to_idx, n=len(G), bot_idx=bot_idx),
        ),
        ("succ2", 2): PredicateClauseCache(
            X1=compile_clause_set_stack(S1_f, C, atom_to_idx, n=len(G), bot_idx=bot_idx),
            X2=compile_clause_set_stack(S2_f, C, atom_to_idx, n=len(G), bot_idx=bot_idx),
        ),
    }

    learner = ProgramLearner(caches)
    print("W shapes:", learner.get_W("even",1).shape, learner.get_W("succ2",2).shape)

    positive_even = [
        Atom("even", ("0",)),
        Atom("even", ("2",)),
        Atom("even", ("4",)),
    ]
    targets_even = build_targets_from_positives(
        atom_to_idx=atom_to_idx,
        constants=C,
        pred_name="even",
        arity=1,
        positive_atoms=positive_even,
    )
    print("even targets:", len(targets_even.pos_idx), "pos,", len(targets_even.neg_idx), "neg")

    a0_t = torch.tensor(a0, dtype=torch.float32)

    # forward paper-like (no training yet)
    aT_pred = learner.infer_T_paper(a0_t, T=Pi.T, temperature=1.0)

    from learning.model import bce_pos_neg, pair_distribution_entropy

    loss = bce_pos_neg(aT_pred, targets_even.pos_idx, targets_even.neg_idx)

    # add small entropy reg on both even and succ2 (semi-latent prior)
    ent_even = pair_distribution_entropy(learner.get_W("even",1), temperature=1.0)
    ent_succ2 = pair_distribution_entropy(learner.get_W("succ2",2), temperature=1.0)
    loss_total = loss + 1e-3 * (ent_even + ent_succ2)

    print("loss_even:", float(loss.item()), "loss_total:", float(loss_total.item()))

    loss_total.backward()
    print("✅ backward OK")


    B1 = [
        Atom("zero", ("0",)),
        Atom("succ", ("0", "1")),
        Atom("succ", ("1", "2")),
        Atom("succ", ("2", "3")),
        Atom("succ", ("3", "4")),
        Atom("succ", ("4", "5")),
    ]

    B2 = [
        Atom("zero", ("0",)),
        Atom("succ", ("0", "1")),
        Atom("succ", ("1", "2")),
        Atom("succ", ("2", "3")),
        Atom("succ", ("3", "4")),
        # niente succ(4,5)
    ]

    a0_1 = build_a0(n=n, atom_to_idx=atom_to_idx, B=B1, bot_idx=bot_idx)
    a0_2 = build_a0(n=n, atom_to_idx=atom_to_idx, B=B2, bot_idx=bot_idx)

    a0_batch = torch.tensor([a0_1, a0_2], dtype=torch.float32)  # [2,n]

    # Generate raw clause sets for all intensional preds in Π
    clause_sets = build_clause_sets_for_program(
        predicates=predicates,
        target_pred=target_pred,
        program=Pi,
    )

    # Define bias (generalizable, not hard-coded in filtering functions)
    bias = BiasConfig(
        allowed_body_preds={
            ("succ2", 2): {"succ"},          # succ2 can only be defined via succ
            ("even", 1): {"zero", "succ2", "even"},  # even can use these
        },
        require_recursive={
            ("even", 1): False,              # we’ll enforce recursion only on C2 via require_recursive_on_C2
        },
        require_body_connected=True,
    )

    caches, clause_texts = build_caches_with_bias(
        clause_sets=clause_sets,
        constants=C,
        atom_to_idx=atom_to_idx,
        n=n,
        bot_idx=bot_idx,
        bias=bias,
        require_recursive_on_C2={
            ("even", 1): True,   # enforce recursion in τ2 for even
            # succ2 doesn't need recursion
        }
    )

    learner = ProgramLearner(caches)

    positive_even = [Atom("even", ("0",)), Atom("even", ("2",)), Atom("even", ("4",))]
    targets_even = build_targets_from_positives(
        atom_to_idx=atom_to_idx,
        constants=C,
        pred_name="even",
        arity=1,
        positive_atoms=positive_even,
    )

    cfg = TrainConfig(epochs=400, lr=5e-2, temperature_start=2.0, temperature_end=0.2, entropy_coeff=1e-3, log_every=50)

    train_program(
        learner=learner,
        a0_batch=a0_batch,
        T=Pi.T,
        target_key=("even", 1),
        targets=targets_even,
        cfg=cfg,
        clause_texts=clause_texts,
    )


if __name__ == "__main__":
    main()