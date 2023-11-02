import argparse
import os

import jax.lax
import jax.numpy as jnp
import jax.random
import numpy as np
from jax import jit, value_and_grad
from jax.tree_util import tree_flatten, tree_unflatten

import legate.jax

NUM_HEADS = 2


def scale_dot_product_attention(q, k, v):
    attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k)
    attn_weights = jax.nn.softmax(attn_weights)
    return jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)


def dense(inputs):
    d_model = inputs.shape[-1]
    kernel = jnp.full((d_model, d_model), 1.0)
    return jax.lax.dot_general(
        inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ()))
    )


def layer(params, x):
    x = dense(x)
    params = dense(params)

    old_shape = x.shape
    new_shape = (
        old_shape[0],
        old_shape[1],
        NUM_HEADS,
        old_shape[2] // NUM_HEADS,
    )
    x = x.reshape(new_shape)
    params = params.reshape(new_shape)

    x = scale_dot_product_attention(x, params, params)
    x = x.reshape(old_shape)

    x = dense(x)

    x = jnp.cos(x)
    return x


layer = legate.jax.task(layer)


def loss_fn(params, x):
    for lyr, param in enumerate(params):
        x = layer(param, x, name=f"layer_{lyr}")
    return legate.jax.task(lambda x: (x * x).sum())(x, name="loss")


def step_fn(batch, params):
    grad_fn = value_and_grad(loss_fn)

    value, grads = grad_fn(params, batch)

    flat_params, tree = tree_flatten(params)
    flat_grads, _ = tree_flatten(grads)
    new_params = []
    for param, grad in zip(flat_params, flat_grads):
        new_params.append(param - 0.05 * grad)
    return value, tree_unflatten(tree, new_params)


def run_step(
    num_layers: int,
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    abstract: bool = False,
):
    shape = (batch_size, seq_length, hidden_size)

    key = jax.random.PRNGKey(42)

    def generator(shape, key):
        return jax.random.uniform(key, shape, np.float32)

    batch = generator(shape, key)
    params = []
    for _ in range(num_layers):
        key, subkey = jax.random.split(key)
        params.append(generator(shape, subkey))

    step = jit(step_fn)
    loss, params = step(batch, params)
    print("Loss=", loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nlayers",
        type=int,
        dest="num_layers",
        default=2,
        help="Number of layers",
    )
    parser.add_argument(
        "--batch",
        type=int,
        dest="batch_size",
        default=1,
        help="Size of microbatch dimension",
    )
    parser.add_argument(
        "--seq",
        type=int,
        dest="seq_length",
        default=2,
        help="Size of sequence dimension",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        dest="hidden_size",
        default=4,
        help="Size of hidden dimension",
    )
    parser.add_argument(
        "--dump-hlo-txt",
        action="store_true",
        dest="dump_hlo_txt",
        default=False,
        help="Dump HLO module text",
    )
    parser.add_argument(
        "--dump-to",
        type=str,
        dest="dump_dir",
        default="dump_microbatches",
        help="Path to dump HLO modules",
    )
    parser.add_argument(
        "--extra",
        dest="extra_xla_flags",
        action="append",
        required=False,
        default=[],
        help="Extra XLA flags",
    )
    parser.add_argument(
        "--abstract",
        dest="abstract",
        action="store_true",
        default=False,
        help="Whether to run abstract evaluation with flops/bytes",
    )

    args, _ = parser.parse_known_args()

    xla_flags = ["--xla_dump_hlo_as_proto"]
    if args.dump_hlo_txt:
        xla_flags.append("--xla_dump_hlo_as_text")
    xla_flags.append(f"--xla_dump_to={args.dump_dir}")
    if len(args.extra_xla_flags) > 0:
        xla_flags += args.extra_xla_flags
    os.environ["XLA_FLAGS"] = " ".join(xla_flags)

    run_step(
        args.num_layers,
        args.batch_size,
        args.seq_length,
        args.hidden_size,
        args.abstract,
    )
