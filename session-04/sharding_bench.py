import jax

from functools import partial

import timing_util

MAXTIX_DIM = 16384

mesh = jax.sharding.Mesh(jax.devices(), 'myaxis')

sharded_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('myaxis'))
unsharded_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))

A = jax.numpy.ones((MAXTIX_DIM, MAXTIX_DIM), dtype=jax.numpy.bfloat16, device=sharded_sharding)
# A = jax.device_put(A, sharded_sharding)

jax.debug.visualize_array_sharding(A)

@partial(jax.jit, out_shardings=unsharded_sharding)
def unshard(input):
    return input

@partial(jax.jit, out_shardings=sharded_sharding)
def shard(input):
    return input

average_time_ms = timing_util.simple_timeit(unshard, A, task='unshard_array')

achieved_bandwidth_GB_s = (A.size * 2 / 1e9) / (average_time_ms / 1e3)

print(f"{achieved_bandwidth_GB_s=}")

A_unsharded = unshard(A)
jax.debug.visualize_array_sharding(A_unsharded)

