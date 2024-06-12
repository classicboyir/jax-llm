import datetime
import jax
from jax import jit
import timing_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

key = jax.random.PRNGKey(0)

MATRIX_DIM = 16384 * 2
STEPS = 10

NUM_MATRICES = 2**28 // MATRIX_DIM ** 2

A = jax.random.normal(key, (MATRIX_DIM, MATRIX_DIM), dtype=jnp.bfloat16)
B = jax.numpy.zeros((MATRIX_DIM, MATRIX_DIM), dtype=jnp.bfloat16)

mesh = jax.sharding.Mesh(jax.devices(), 'myaxis')

sharding_shards = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('myaxis'))

Ones_shard = jax.device_put(A, sharding_shards)
Zeros_shard = jax.device_put(B, sharding_shards)

@partial(jax.jit, out_shardings=sharding_shards)
def unshard(input):
    return input

# @jax.jit
@partial(jax.jit, out_shardings=sharding_shards)
def f(A, B):
    return jax.lax.batch_matmul(A, B)


@partial(jax.jit, out_shardings=sharding_shards)
def matmul_normal(A, B):
    return A @ B


def calculate_time(input_A, average_time_ms):
    achieved_bandwidth_GB_s = (input_A.size * 2 / 1e9) / (average_time_ms / 1e3)
    print(f"{achieved_bandwidth_GB_s=}, {average_time_ms=}")


calculate_time(Ones_shard, timing_util.simple_timeit(f, Ones_shard, Ones_shard, task='unshard_array'))

calculate_time(Zeros_shard, timing_util.simple_timeit(f, Zeros_shard, Zeros_shard, task='unshard_array'))

calculate_time(Ones_shard, timing_util.simple_timeit(matmul_normal, Ones_shard, Ones_shard, task='normal_matmul_ones'))

calculate_time(Zeros_shard, timing_util.simple_timeit(matmul_normal, Zeros_shard, Zeros_shard, task='normal_matmul_zeros'))


