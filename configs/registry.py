from __future__ import annotations

from typing import Callable, Dict
from learning.task_config import TaskConfig

from configs.toy_even import make_config as make_toy_even


TASK_CONFIGS: Dict[str, Callable[[], TaskConfig]] = {
    "toy_even": make_toy_even,
}