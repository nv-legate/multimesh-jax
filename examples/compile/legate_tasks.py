import re
from dataclasses import dataclass
from typing import Optional

import gin


@gin.configurable
@dataclass
class PaxTransformerConfig:
    transformer_num_devices: int = 4
    logits_num_devices: int = 4
    embeddings_num_devices: int = 4
    layers_per_stage: int = 1
    layers_per_interleave: Optional[int] = None

    def __call__(self):
        import jax

        from legate.jax import register_task, register_task_factory

        print("Devices = ", jax.devices())
        num_devices = 32  # len(jax.devices())
        devices = list(range(num_devices))

        layer_regex = re.compile(r"layers_(\d+)")

        def compute_devices(name: str):
            layer = int(layer_regex.search(name).groups()[0])
            if self.layers_per_interleave is not None:
                # layer offset within an interleave
                layer = layer % self.layers_per_interleave
            stage = layer // self.layers_per_stage
            offset = self.transformer_num_devices * stage
            stop = offset + self.transformer_num_devices
            return list(range(offset, stop))

        register_task_factory(
            r"(layers_\d+)",
            device_callback=compute_devices,
            dims=[self.transformer_num_devices, 1],
            device_axes=["x", "y"],
            logical_axes=[
                ("replica", "y"),
                ("mdl", "x"),
                ("data", "y"),
            ],
        )

        register_task(
            "(emb_lookup).*",
            devices=devices[: self.embeddings_num_devices],
            dims=[self.embeddings_num_devices, 1],
            device_axes=["x", "y"],
            logical_axes=[
                ("replica", "y"),
                ("mdl", "x"),
                ("data", "y"),
            ],
        )

        register_task(
            "(position_emb).*",
            devices=devices[: self.embeddings_num_devices],
            dims=[self.embeddings_num_devices, 1],
            device_axes=["x", "y"],
            logical_axes=[
                ("replica", "y"),
                ("mdl", "x"),
                ("data", "y"),
            ],
        )

        register_task(
            "(final_ln).*",
            devices=devices[-self.logits_num_devices :],
            dims=[self.logits_num_devices, 1],
            device_axes=["x", "y"],
            logical_axes=[
                ("replica", "x"),
                ("mdl", "x"),
                ("data", "y"),
            ],
        )

        register_task(
            "(compute_loss).*",
            devices=devices[-self.logits_num_devices :],
            dims=[self.logits_num_devices, 1],
            device_axes=["x", "y"],
            logical_axes=[
                ("replica", "x"),
                ("mdl", "x"),
                ("data", "y"),
            ],
        )

        register_task(
            "default",
            devices=devices[: self.embeddings_num_devices],
            dims=[self.embeddings_num_devices, 1],
            device_axes=["x", "y"],
            logical_axes=[
                ("replica", "x"),
                ("mdl", "x"),
                ("data", "y"),
            ],
        )
