from __future__ import annotations

from configs.mnist_addition import make_config
from learning.system_builder import build_system_from_config
from data.mnist_addition_adapter import MNISTAdditionStubAdapter
import torch
from logic.atoms import Atom
from logic.valuation_soft import build_a0_from_indexed_facts
from learning.data import build_targets_from_positives

def test_mnist_addition_pipeline_smoke():
    cfg = make_config()
    print("Language constants:", cfg.constants)
    bundle = build_system_from_config(cfg)
    print("Language predicates:", cfg.predicates)

    assert len(bundle.G) > 0
    assert 0 <= bundle.bot_idx < len(bundle.idx_to_atom)
    assert ("sum_is", 1) in bundle.learner.caches

def test_mnist_addition_adapter_stub_smoke():
    cfg = make_config()
    bundle = build_system_from_config(cfg)
    adapter = MNISTAdditionStubAdapter(num_examples=4, noise=0.1, seed=123, supervised_concepts=True)
    examples = adapter.build_examples(atom_to_idx=bundle.atom_to_idx)

    assert len(examples) == 4

    ex0 = examples[0]

    # soft_facts should include 20 digit atoms (d1:10 + d2:10)
    digit_facts = [a for (a, p) in ex0.soft_facts if a.pred == "digit"]
    assert len(digit_facts) == 20

    # task target is sum_is/1 with 1 positive and 18 negatives
    assert len(ex0.task_targets.pos_idx) == 1
    assert len(ex0.task_targets.neg_idx) == 18

    # concept targets exist in supervised mode
    assert ex0.concept_targets is not None
    assert "digit_d1" in ex0.concept_targets and "digit_d2" in ex0.concept_targets
    assert len(ex0.concept_targets["digit_d1"].pos_idx) == 1
    assert len(ex0.concept_targets["digit_d1"].neg_idx) == 9

def test_mnist_addition_end_to_end_backward_smoke():
    cfg = make_config()
    bundle = build_system_from_config(cfg)

    # ---- CBM stub: create differentiable logits for two digits ----
    torch.manual_seed(0)
    logits1 = torch.randn(10, requires_grad=True)
    logits2 = torch.randn(10, requires_grad=True)

    probs1 = torch.softmax(logits1, dim=0)  # [10]
    probs2 = torch.softmax(logits2, dim=0)  # [10]

    # Pretend the true digits are:
    d1_true = 3
    d2_true = 5
    sum_true = d1_true + d2_true  # 8

    # ---- Build indexed soft facts for digit(d1,d) and digit(d2,d) ----
    soft_atoms = []
    soft_vals = []

    for d in range(10):
        soft_atoms.append(Atom("digit", ("d1", str(d))))
        soft_vals.append(probs1[d])
    for d in range(10):
        soft_atoms.append(Atom("digit", ("d2", str(d))))
        soft_vals.append(probs2[d])

    soft_idx = torch.tensor([bundle.atom_to_idx[a] for a in soft_atoms], dtype=torch.long)
    soft_val = torch.stack(soft_vals).to(dtype=torch.float32)  # [20]

    # No hard facts for now
    a0 = build_a0_from_indexed_facts(
        n=len(bundle.G),
        bot_idx=bundle.bot_idx,
        soft_idx=soft_idx,
        soft_val=soft_val,
        hard_idx=None,
    )

    # ---- ILP inference ----
    aT = bundle.learner.infer_T_paper(a0, T=bundle.program.T, temperature=1.0, fast=True)

    # ---- Task targets: sum_is(sum_true) ----
    sums = [str(s) for s in range(19)]
    targets = build_targets_from_positives(
        atom_to_idx=bundle.atom_to_idx,
        constants=cfg.constants,
        pred_name="sum_is",
        arity=1,
        positive_atoms=[Atom("sum_is", (str(sum_true),))],
    )

    # BCE loss on sum_is/1 atoms
    from learning.model import bce_pos_neg
    loss_task = bce_pos_neg(aT, targets.pos_idx, targets.neg_idx)

    # ---- Concept supervised loss (CBM supervised phase) ----
    # negative log-likelihood of true digits
    eps = 1e-9
    loss_concepts = -(torch.log(probs1[d1_true] + eps) + torch.log(probs2[d2_true] + eps)) / 2.0

    # Joint loss (supervised CBM phase)
    loss = loss_task + 0.5 * loss_concepts

    loss.backward()

    assert logits1.grad is not None
    assert logits2.grad is not None

if __name__ == "__main__":
    test_mnist_addition_pipeline_smoke()
    print("✅ tests/test_mnist_addition_pipeline.py passed (part 1)")
    test_mnist_addition_adapter_stub_smoke()
    print("✅ tests/test_mnist_addition_pipeline.py passed (part 2)")
    test_mnist_addition_end_to_end_backward_smoke()
    print("✅ tests/test_mnist_addition_pipeline.py passed (parts 3)")