import unittest

import gin
import numpy as np
from google.protobuf import text_format
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.service import hlo_pb2

from lllm.mesh import TaskMesh, legate_global_mesh

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
  "num_layers" : %NUM_LAYERS,
  "device_axes" : ('ax',),
  "logical_axes" : %AXES,
  "device_shape" : %TASK_MESH,
}}

legate_global_mesh:
  mesh_args = [
    (@mesh.LayerMesh, "layer_{{}}.*", (%ARGS,)),
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
  "num_layers" : {num_layers},
  "num_interleaves" : {num_interleaves},
  "device_axes" : ('ax',),
  "logical_axes" : %AXES,
  "device_shape" : %TASK_MESH,
}}

legate_global_mesh:
  mesh_args = [
    (@mesh.InterleavedLayerMesh, "layer_{{}}.*", (%ARGS,)),
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
  "device_axes" : ('ax',),
  "logical_axes" : %AXES,
  "device_shape" : %TASK_MESH,
}}

legate_global_mesh:
  mesh_args = [
    (@mesh.TaskMesh, "layer.*", (%ARGS,)),
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

    def test_permuted_axes_full_replication(self):
        logical_axes = [
            ("data", "x"),
            ("model", "y"),
        ]
        model = 4
        data = 2
        devices = np.arange(data * model).reshape(data, model)
        mesh = TaskMesh(
            matcher="n/a",
            device_axes=("x", "y"),
            logical_axes=logical_axes,
            device_shape=(data, model),
            devices=devices,
        )
        instr = hlo_pb2.HloInstructionProto()
        instr.shape.dimensions.extend([64, 128, 256])
        axes = ("data", "model", "data")
        instr.metadata.op_name += f"/legate_axes={axes}/"

        mesh.shard(instr)

        self.assertEqual(
            np.prod(instr.sharding.tile_assignment_dimensions), model * data
        )
        self.assertEqual(instr.sharding.tile_assignment_dimensions[-1], data)
        self.assertFalse(instr.sharding.replicate_on_last_tile_dim)

        # 1st and 2nd axes should get permuted here
        correct = [0, 4, 1, 5, 2, 6, 3, 7]
        np.testing.assert_equal(
            instr.sharding.tile_assignment_devices, correct
        )

    def test_partial_replication(self):
        logical_axes = [
            ("data", "x"),
            ("model", "y"),
        ]
        model = 4
        data = 2
        devices = np.arange(data * model).reshape(data, model)
        mesh = TaskMesh(
            matcher="n/a",
            device_axes=("x", "y"),
            logical_axes=logical_axes,
            device_shape=(data, model),
            devices=devices,
        )
        instr = hlo_pb2.HloInstructionProto()
        instr.shape.dimensions.extend([64, 128, 256])
        axes = (None, None, "model")
        instr.metadata.op_name = f"/legate_axes={axes}/"

        mesh.shard(instr)

        self.assertEqual(
            np.prod(instr.sharding.tile_assignment_dimensions), model * data
        )
        self.assertEqual(instr.sharding.tile_assignment_dimensions[-1], data)
        self.assertTrue(instr.sharding.replicate_on_last_tile_dim)
        self.assertEqual(
            len(instr.sharding.tile_assignment_dimensions), len(axes) + 1
        )

        # 1st and 2nd axes should get permuted here
        correct = [0, 4, 1, 5, 2, 6, 3, 7]
        # [ 0 4 ]
        # [ 1 5 ]
        # [ 2 6 ]
        # [ 3 7 ]
        # Tile [0,0,0,0]=0 -> [0,0,0]=0
        # Tile [0,0,0,1]=1 -> [0,0,0]=0
        # Tile [0,0,1,0]=2 -> [0,0,1]=1
        # Tile [0,0,1,1]=3 -> [0,0,1]=1
        # Devices 0, 2, 4, 6 should have one set of replicas
        # Devices 1, 3, 5, 7 should the other set of replicas
        np.testing.assert_equal(
            instr.sharding.tile_assignment_devices, correct
        )

        axes = (None, "data", None)
        instr.metadata.op_name = f"/legate_axes={axes}/"

        mesh.shard(instr)

        self.assertEqual(
            np.prod(instr.sharding.tile_assignment_dimensions), model * data
        )
        self.assertEqual(instr.sharding.tile_assignment_dimensions[-1], model)
        self.assertTrue(instr.sharding.replicate_on_last_tile_dim)
        self.assertEqual(
            len(instr.sharding.tile_assignment_dimensions), len(axes) + 1
        )

        # the model (y-axis) is the replicating axis
        # Tile [0,0,0,0]=0 -> [0,0,0]=0
        # Tile [0,0,0,1]=1 -> [0,0,0]=0
        # Tile [0,0,0,2]=2 -> [0,0,0]=0
        # Tile [0,0,0,3]=3 -> [0,0,0]=0
        # Tile [0,1,0,0]=4 -> [0,0,0]=1
        # Tile [0,1,0,1]=5 -> [0,0,0]=1
        # Tile [0,1,0,2]=6 -> [0,0,0]=1
        # Tile [0,1,0,3]=7 -> [0,0,0]=1
        # Devices 0, 4 should have one set of replicas
        # Devices 1, 5 should have one set of replicas, etc
        correct = [0, 1, 2, 3, 4, 5, 6, 7]
        # [ 0 1 ]
        # [ 2 3 ]
        # [ 4 5 ]
        # [ 6 7 ]
        np.testing.assert_equal(
            instr.sharding.tile_assignment_devices, correct
        )

    def test_multidevice_axes(self):
        logical_axes = [("data", "x"), ("data", "z"), ("model", "y")]
        x = 2
        y = 1
        z = 3
        devices = np.arange(x * y * z).reshape(x, y, z)
        mesh = TaskMesh(
            matcher="n/a",
            device_axes=("x", "y", "z"),
            logical_axes=logical_axes,
            device_shape=(x, y, z),
            devices=devices,
        )
        instr = hlo_pb2.HloInstructionProto()
        instr.shape.dimensions.extend([64, 128, 256])
        axes = (None, "data", "model")
        instr.metadata.op_name = f"/legate_axes={axes}/"

        mesh.shard(instr)
        np.testing.assert_equal(
            instr.sharding.tile_assignment_dimensions, [1, 6, 1]
        )
        np.testing.assert_equal(
            instr.sharding.tile_assignment_devices, [0, 1, 2, 3, 4, 5]
        )


if __name__ == "__main__":
    unittest.main()
