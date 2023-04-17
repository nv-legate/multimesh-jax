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

NUM_HEADS = 4


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


def layer(lyr, params, x):
    x = partition(x, "x", "y", "z")
    params = partition(params, "x", "y", "z")

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
    return partition(x, "x")


def loss_fn(params, x):
    for lyr, param in enumerate(params):
        with task(name="layer", layer=lyr, backprop=True):
            for _ in range(NUM_REPS):
                x = layer(lyr, param, x)
    with task(
        append=True, backprop=True
    ):  # add the loss computation to the last layer
        return (x * x).sum()


def step_fn(microbatches, params):
    grad_fn = value_and_grad(loss_fn)
    # simulate unrolled microbatching

    def body_fun(i, args):
        microbatches, params = args
        batch = microbatches[i]
        return grad_fn(params, batch)

    reduce = reduce_body(lambda x, y: x + y)
    reduce_init = (0, pytree_zeros(params))
    num_microbatches = len(microbatches)
    value, grads = fori_reduce(
        0,
        num_microbatches,
        name="microbatch",
        args=(microbatches, params),
        body_fun=body_fun,
        reduce_init=reduce_init,
        reduce=reduce,
        implicit_decomposition=True,
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
    num_reps: int,
    num_microbatches: int,
    microbatch_size: int,
    seq_size: int,
    hidden_size: int,
    abstract: bool = False,
):
    shape = (microbatch_size, seq_size, hidden_size)

    def generator(shape):
        if abstract:
            return AbstractTensor(shape, dtype=np.float32)
        else:
            return np.zeros(shape, np.float32)

    microbatches = [generator(shape) for _ in range(num_microbatches)]
    params = [generator(shape) for _ in range(num_layers)]

    step = jit(step_fn)
    loss, params = step(microbatches, params)
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
        "--nreps",
        type=int,
        dest="num_reps",
        default=4,
        help="Number of repetitions in each layer",
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
        default=256,
        help="Size of microbatch dimension",
    )
    parser.add_argument(
        "--seq",
        type=int,
        dest="seq_size",
        default=1024,
        help="Size of sequence dimension",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        dest="hidden_size",
        default=512,
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

    NUM_REPS = args.num_reps

    run_step(
        args.num_layers,
        args.num_reps,
        args.num_microbatches,
        args.microbatch_size,
        args.seq_size,
        args.hidden_size,
        args.abstract,
    )
