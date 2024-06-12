import jax
import timing_util
import jax.numpy as jnp
from functools import partial

MAXRIX_SIZE = 16384
BATCH_PER_CHIP = 4096

ACTIVATION = jax.numpy.ones((BATCH_PER_CHIP * jax.device_count(), MAXRIX_SIZE), dtype=jnp.bfloat16)
W = jax.numpy.ones((MAXRIX_SIZE, MAXRIX_SIZE), dtype=jnp.bfloat16)

mesh = jax.sharding.Mesh(jax.devices(), ("myaxis"))

activation_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("myaxis", None))
weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "myaxis"))

ACTIVATION = jax.device_put(ACTIVATION, activation_sharding)
W = jax.device_put(W, activation_sharding)

@jax.jit
def matmul_jit(A, B):
    return A @ B

average_time_ms = timing_util.simple_timeit(matmul_jit, ACTIVATION, W, task='all_gather_shard')

