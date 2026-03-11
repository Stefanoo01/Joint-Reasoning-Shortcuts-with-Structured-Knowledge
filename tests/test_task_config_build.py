from __future__ import annotations

from configs.toy_even import make_config
from ilp.learning.system_builder import build_system_from_config


def test_build_system_from_config():
    cfg = make_config()
    bundle = build_system_from_config(cfg)

    assert len(bundle.G) > 0
    assert 0 <= bundle.bot_idx < len(bundle.idx_to_atom)
    assert bundle.idx_to_atom[bundle.bot_idx].pred == "__bot__"
    assert ("even", 1) in bundle.learner.caches
    assert ("succ2", 2) in bundle.learner.caches


if __name__ == "__main__":
    test_build_system_from_config()
    print("✅ tests/test_task_config_build.py passed")