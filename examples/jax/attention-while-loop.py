import argparse
import os

import jax.lax
import jax.nn
import jax.numpy as jnp
import jax.random
import numpy as np
from jax import jit, value_and_grad
from jax.tree_util import tree_flatten, tree_unflatten

from lllm.api import (
    fori_reduce,
    mark_gradient_scalar,
    partition,
    pytree_zeros,
    reduce_body,
    task,
)
from lllm.api.abstract import AbstractTensor


def scale_dot_product_attention(q, k, v):
    attn_weights = jnp.einsum("bqhd,khd->bhqk", q, k)
    attn_weights = jax.nn.softmax(attn_weights)
    return jnp.einsum("bhqk,khd->bqhd", attn_weights, v)


def dense(inputs):
    d_model = inputs.shape[-1]
    kernel = jnp.full((d_model, d_model), 1.0)
    return jax.lax.dot_general(
        inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ()))
    )


def project(x, num_heads: int):
    new_shape = list(x.shape[:-1]) + [num_heads, x.shape[-1] // num_heads]
    return x.reshape(new_shape)


def layer(params, x, num_heads: int):
    x = partition(x, "batch", "seq", "embed")
    params = partition(params, "key", "heads", "hidden")

    x = dense(x)
    input_shape = x.shape

    x = project(x, num_heads)  # "batch", "seq", "heads", "hidden"
    x = scale_dot_product_attention(x, params, params)
    x = x.reshape(input_shape)
    x = dense(x)

    x = jnp.cos(x)
    return partition(x, "batch", "seq", "embed")


def loss_fn(params, x, num_heads: int):
    for lyr, param in enumerate(params):
        with task(name="layer", layer=lyr, backprop=True):
            x = layer(param, x, num_heads)
    with task(name="loss", backprop=True):
        return (x * x).sum()


def step_fn(batch, params, num_heads: int, num_microbatches: int, microbatch_size: int):
    grad_fn = value_and_grad(loss_fn)

    def body_fun(i, args):
        batch, params = args
        offset = i * microbatch_size
        starts = [offset] + [0] * (batch.ndim - 1)
        lengths = [microbatch_size] + list(batch.shape[1:])
        print(num_microbatches, microbatch_size, starts, lengths)
        microbatch = jax.lax.dynamic_slice(batch, starts, lengths)
        print("microbatch", microbatch.shape)
        return grad_fn(params, microbatch, num_heads)

    def reduce_fun(i,x,y):
        x_flat, treedef = tree_flatten(x)
        y_flat, _ = tree_flatten(y)
        result = [ x+y for (x,y) in zip(x_flat,y_flat) ]
        return tree_unflatten(treedef, result)

    reduce_init = (0, pytree_zeros(params))
    #num_microbatches = len(microbatches)
    value, grads = fori_reduce(
        0,
        num_microbatches,
        name="microbatch",
        args=(batch, params),
        body_fun=body_fun,
        reduce_init=reduce_init,
        reduce=reduce_fun,
        implicit_decomposition=True,
        unroll = False
    )

    flat_params, tree = tree_flatten(params)
    flat_grads, _ = tree_flatten(grads)
    new_params = []
    with task(name="update", implicit_decomposition=True):
        for param, grad in zip(flat_params, flat_grads):
            new_params.append(param - 0.05 * grad)
    return mark_gradient_scalar(value), tree_unflatten(tree, new_params)


def run_step(
    num_layers: int,
    num_microbatches: int,
    microbatch_size: int,
    seq_size: int,
    hidden_size: int,
    num_heads: int,
    abstract: bool = False,
):
    embed_size = num_heads * hidden_size
    input_shape = (microbatch_size * num_microbatches, seq_size, embed_size)
    param_shape = (seq_size, num_heads, hidden_size)

    def generator(shape):
        if abstract:
            return AbstractTensor(shape, dtype=np.float32)
        else:
            return np.zeros(shape, np.float32)

    #microbatches = [generator(input_shape) for _ in range(num_microbatches)]
    batch = generator(input_shape)
    params = [generator(param_shape) for _ in range(num_layers)]

    step = jit(step_fn, static_argnums=[2,3,4])
    loss, params = step(batch, params, num_heads, num_microbatches, microbatch_size)
    print("Loss=", loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nlayers",
        type=int,
        dest="num_layers",
        default=4,
        help="Number of layers",
    )
    parser.add_argument(
        "--nbatch",
        type=int,
        dest="num_microbatches",
        default=4,
        help="Number of microbatches",
    )
    parser.add_argument(
        "--batch",
        type=int,
        dest="microbatch_size",
        default=32,
        help="Size of microbatch dimension",
    )
    parser.add_argument(
        "--seq",
        type=int,
        dest="seq_size",
        default=512,
        help="Size of sequence dimension",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        dest="hidden_size",
        default=64,
        help="Size of hidden dimension",
    )
    parser.add_argument(
        "--heads",
        type=int,
        dest="heads",
        default=4,
        help="No. of attention heads",
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
    print(xla_flags)
    os.environ["XLA_FLAGS"] = " ".join(xla_flags)

    run_step(
        args.num_layers,
        args.num_microbatches,
        args.microbatch_size,
        args.seq_size,
        args.hidden_size,
        args.heads,
        args.abstract,
    )
