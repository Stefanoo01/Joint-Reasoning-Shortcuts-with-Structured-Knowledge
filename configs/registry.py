from __future__ import annotations

from typing import Callable, Dict
from ilp.learning.task_config import TaskConfig

from configs.toy_even import make_config as make_toy_even
from configs.toy_sum_parity import make_config as make_toy_sum_parity


TASK_CONFIGS: Dict[str, Callable[[], TaskConfig]] = {
    "toy_even": make_toy_even,
    "toy_sum_parity": make_toy_sum_parity,
}