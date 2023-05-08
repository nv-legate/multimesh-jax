import unittest

import gin
from google.protobuf import text_format
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.service import hlo_pb2

from lllm.mesh import legate_global_mesh

TEST_HLO_PROTO = """shape {
  dimensions: 16
  dimensions: 16
}
metadata {
  op_name: "legate_axes=(\'data\', \'model\')/"
}"""

NUM_LAYERS = 2
TASK_MESH_SIZE = 4


LAYER_MESH_STRING = """
NUM_LAYERS = {num_layers}
AXES = (
  ("data", "ax"),
  ("model", "ax"),
)
TASK_MESH = ({size},)

ARGS = {{
  "matcher_template" : "layer_{{}}.*",
  "num_layers" : %NUM_LAYERS,
  "device_axes" : ('ax',),
  "logical_axes" : %AXES,
  "device_shape" : %TASK_MESH,
}}

legate_global_mesh:
  mesh_args = [
    (@mesh.LayerMesh, %ARGS),
  ]
""".format(  # noqa: E501
    num_layers=NUM_LAYERS, size=TASK_MESH_SIZE
)

NUM_INTERLEAVES = 2
_INTERLEAVED_MESH_STRING = """
AXES = (
  ("data", "ax"),
  ("model", "ax"),
)
TASK_MESH = ({size},)

ARGS = {{
  "matcher_template" : "layer_{{}}.*",
  "num_layers" : {num_layers},
  "num_interleaves" : {num_interleaves},
  "device_axes" : ('ax',),
  "logical_axes" : %AXES,
  "device_shape" : %TASK_MESH,
}}

legate_global_mesh:
  mesh_args = [
    (@mesh.InterleavedLayerMesh, %ARGS),
  ]
"""  # noqa: E501
INTERLEAVED_MESH_STRING = _INTERLEAVED_MESH_STRING.format(
    num_layers=NUM_LAYERS * NUM_INTERLEAVES,
    size=TASK_MESH_SIZE,
    num_interleaves=NUM_INTERLEAVES,
    num_layers_per_task=1,
)


TASK_MESH_STRING = """
NUM_LAYERS = 2
AXES = (
  ("data", "ax"),
  ("model", "ax"),
)
TASK_MESH = ({size},)

ARGS = {{
  "matcher" : "layer.*",
  "device_axes" : ('ax',),
  "logical_axes" : %AXES,
  "device_shape" : %TASK_MESH,
}}

legate_global_mesh:
  mesh_args = [
    (@mesh.TaskMesh, %ARGS),
  ]
""".format(
    size=TASK_MESH_SIZE
)

TASK_MESH_SHARDING = """type: OTHER
tile_shape {
  dimensions: 16
  dimensions: 16
}
tile_assignment_dimensions: 4
tile_assignment_dimensions: 1
tile_assignment_devices: 0
tile_assignment_devices: 1
tile_assignment_devices: 2
tile_assignment_devices: 3"""


class MeshTest(unittest.TestCase):
    def assert_proto_equals(self, lhs, rhs) -> None:
        self.assertEqual(
            text_format.MessageToString(lhs), text_format.MessageToString(rhs)
        )

    def _test_common_task_mesh(self, mesh):
        self.assertEqual(mesh.ndevices, TASK_MESH_SIZE)

        instr = hlo_pb2.HloInstructionProto()
        instr.metadata.op_name = "legate_axes=('data', 'model')/"
        instr.shape.dimensions.append(16)
        instr.shape.dimensions.append(16)

        instr = text_format.Parse(
            TEST_HLO_PROTO, hlo_pb2.HloInstructionProto()
        )
        mesh.shard(instr)

        sharding = text_format.Parse(
            TASK_MESH_SHARDING, xla_data_pb2.OpSharding()
        )

        self.assert_proto_equals(instr.sharding, sharding)

    def test_basic_task_mesh(self):
        gin.clear_config()
        gin.parse_config(TASK_MESH_STRING)
        global_mesh = legate_global_mesh()
        mesh = global_mesh.get_task_mesh("layer_0")
        self._test_common_task_mesh(mesh)

    def test_layer_mesh(self):
        gin.clear_config()
        gin.parse_config(LAYER_MESH_STRING)
        global_mesh = legate_global_mesh()

        for lyr in range(NUM_LAYERS):
            mesh = global_mesh.get_task_mesh(f"layer_{lyr}")
            offset = TASK_MESH_SIZE * lyr
            devices = [offset + i for i in range(TASK_MESH_SIZE)]
            self.assertSequenceEqual(devices, list(mesh.devices))
            # The devices are always given relative numbering from zero
            # We can reuse a common testing routine
            self._test_common_task_mesh(mesh)

    def test_interleaved_mesh(self):
        gin.clear_config()
        gin.parse_config(INTERLEAVED_MESH_STRING)
        global_mesh = legate_global_mesh()
        total_layers = NUM_INTERLEAVES * NUM_LAYERS
        for lyr in range(total_layers):
            mesh = global_mesh.get_task_mesh(f"layer_{lyr}")
            # The devices are always given relative numbering from zero
            # We can reuse a common testing routine
            self._test_common_task_mesh(mesh)
            offset = (lyr % NUM_LAYERS) * TASK_MESH_SIZE
            devices = [offset + i for i in range(TASK_MESH_SIZE)]
            self.assertSequenceEqual(devices, list(mesh.devices))


if __name__ == "__main__":
    unittest.main()
