import unittest

import gin

from lllm.key_utils import KeyTransform, LegateKey, map_legate_key


@gin.configurable
class HelloTransform(KeyTransform):
    def __init__(self):
        pass

    def __call__(self, key: LegateKey) -> LegateKey:
        return LegateKey("hello")


_HELLO_CONFIG_STRING = r"""
map_legate_key:
  transforms = [
    (@HelloTransform, ()),
  ]
"""

_NONE_NAME_CONFIG_STRING = r"""
map_legate_key:
  transforms = [
    (@key_utils.NameTransform, (r'(\s+)(.*)', None, None))
  ]
"""


@gin.configurable
def divide_index(index: str):
    return int(index) // 4


_DIVIDE_INDEX_CONFIG_STRING = r"""
map_legate_key:
  transforms = [
    (@key_utils.NameTransform, (r'([a-z]+)(\d+)', None, @divide_index))
  ]
"""


_EXTERNAL_TRANSFORM_CONFIG_STRING = r"""
from lllm import key_utils_test_transforms

key_utils_test_transforms.CompressLayers:
  num_layers_per_task = 2

map_legate_key:
  transforms = [
    (@key_utils_test_transforms.CompressLayers, ()),
  ]
"""

_COMPRESS_CONFIG_STRING = r"""
key_utils.CompressLayersTransform:
  num_layers = {num_layers}
  num_procs = {num_procs}
  num_interleaves = {num_interleaves}

map_legate_key:
  transforms = [
    (@key_utils.CompressLayersTransform, ())
  ]
"""

_COMBINE_LAYERS_STRING = r"""
to_combine = {
  "combined13" : ["layer1", "layer3"],
  "combined2" : ["layer2"],
}

key_utils.CombineNamesTransform:
  to_combine = %to_combine

map_legate_key:
  transforms = [
    (@key_utils.CombineLayersTransform, ())
  ]
"""


class KeyUtilsTest(unittest.TestCase):
    def test_hello_map_legate_key(self):
        gin.clear_config()
        gin.parse_config(_HELLO_CONFIG_STRING)
        test_key = LegateKey(name="input", task_id=0)
        new_key = map_legate_key(test_key)
        self.assertEqual(new_key, LegateKey(name="hello", task_id=None))

    def test_none_name_map(self):
        gin.clear_config()
        gin.parse_config(_NONE_NAME_CONFIG_STRING)
        test_key = LegateKey(name="input", task_id=0)
        new_key = map_legate_key(test_key)
        self.assertEqual(new_key, test_key)

    def test_divide_index_name_map(self):
        gin.clear_config()
        gin.parse_config(_DIVIDE_INDEX_CONFIG_STRING)
        input_key = LegateKey(name="input34", task_id=0)
        test_key = map_legate_key(input_key)
        # this should divce by 4
        correct_key = input_key.replace(name="input8")
        self.assertEqual(correct_key, test_key)

    def test_external_transform(self):
        gin.clear_config()
        gin.parse_config(_EXTERNAL_TRANSFORM_CONFIG_STRING)
        input_key = LegateKey(name="input_42_suffix", task_id=0)
        test_key = map_legate_key(input_key)
        correct_key = input_key.replace(name="input_21_suffix")
        self.assertEqual(correct_key, test_key)

    def test_compress_layers(self):
        num_layers = 8
        num_procs = 4
        num_interleaves = 1

        gin.clear_config()
        gin.parse_config(
            _COMPRESS_CONFIG_STRING.format(
                num_layers=num_layers,
                num_procs=num_procs,
                num_interleaves=num_interleaves,
            )
        )
        num_per = num_layers // num_procs
        for lyr in range(num_layers):
            input_key = LegateKey(name=f"input_{lyr}")
            test_key = map_legate_key(input_key)
            correct_key = LegateKey(name=f"input_{lyr // num_per}")
            self.assertEqual(correct_key, test_key)

        num_interleaves = 2
        gin.clear_config()
        gin.parse_config(
            _COMPRESS_CONFIG_STRING.format(
                num_layers=num_layers,
                num_procs=num_procs,
                num_interleaves=num_interleaves,
            )
        )
        num_per = num_layers // num_procs // num_interleaves
        for lyr in range(num_layers):
            input_key = LegateKey(name=f"input_{lyr}")
            test_key = map_legate_key(input_key)
            correct_key = LegateKey(name=f"input_{lyr // num_per}")
            self.assertEqual(correct_key, test_key)

        num_procs = 2
        num_interleaves = 2
        gin.clear_config()
        gin.parse_config(
            _COMPRESS_CONFIG_STRING.format(
                num_layers=num_layers,
                num_procs=num_procs,
                num_interleaves=num_interleaves,
            )
        )
        num_per = num_layers // num_procs // num_interleaves
        for lyr in range(num_layers):
            input_key = LegateKey(name=f"input_{lyr}")
            test_key = map_legate_key(input_key)
            correct_key = LegateKey(name=f"input_{lyr // num_per}")
            self.assertEqual(correct_key, test_key)

    def test_combine_layers(self):
        gin.clear_config()
        gin.parse_config(_COMBINE_LAYERS_STRING)
        for name in "layer1", "layer3":
            test_key = LegateKey(name=name, task_id=0)
            correct_key = LegateKey(name="combined13", task_id=0)
            new_key = map_legate_key(test_key)
            self.assertEqual(correct_key, new_key)
        for name in ("layer2",):
            correct_key = LegateKey(name="combined2", task_id=0)
            test_key = LegateKey(name=name, task_id=0)
            new_key = map_legate_key(test_key)
            self.assertEqual(correct_key, new_key)
        for name in ("layer4",):
            test_key = LegateKey(name=name, task_id=0)
            new_key = map_legate_key(test_key)
            # this should be unchanged
            self.assertEqual(test_key, new_key)


if __name__ == "__main__":
    unittest.main()
