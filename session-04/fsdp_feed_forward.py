import jax
import timing_util
import jax.numpy as jnp
from functools import partial

MAXRIX_SIZE = 16384
BATCH_PER_CHIP = 4096
LAYERS = 4

mesh = jax.sharding.Mesh(jax.devices(), ("myaxis"))

activation_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("myaxis", None))
weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "myaxis"))

ACTIVATION = jax.numpy.ones((BATCH_PER_CHIP * jax.device_count(), MAXRIX_SIZE), dtype=jnp.bfloat16, device=activation_sharding)
Ws = [jax.numpy.ones((MAXRIX_SIZE, MAXRIX_SIZE), dtype=jnp.bfloat16, device=weight_sharding) for i in range(LAYERS)]

# ACTIVATION = jax.device_put(ACTIVATION, activation_sharding)
# W = jax.device_put(W, activation_sharding)

jax.debug.visualize_array_sharding(ACTIVATION)

@jax.jit
def matmul_jit(A, Ws):
    for W in Ws:
        A = A @ W

    return A

average_time_ms = timing_util.simple_timeit(matmul_jit, ACTIVATION, Ws, task='fsdp_feed_f')

achieved_bandwidth_GB_s = (ACTIVATION.size * 2 / 1e9) / (average_time_ms / 1e3)

# print(f"{achieved_bandwidth_GB_s=}")

# export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"


