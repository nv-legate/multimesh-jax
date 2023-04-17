import re
from dataclasses import dataclass

import gin

from lllm.key_utils import KeyTransform, LegateKey


@gin.configurable
@dataclass(frozen=True)
class CompressLayers(KeyTransform):
    num_layers_per_task: int = 1
    matcher: str = r"([a-zA-Z_]+)(\d+)(.*)"

    def __call__(self, key: LegateKey) -> LegateKey:
        matcher = re.compile(self.matcher)
        match = matcher.search(key.name)
        if match:
            prefix, index, suffix = match.groups()
            new_index = int(index) // self.num_layers_per_task
            new_name = f"{prefix}{new_index}{suffix}"
            return LegateKey(name=new_name, task_id=key.task_id)
        return key
