import jax
import numpy as np
from functools import partial

MATRIX_SIZE = 16384

A = jax.numpy.ones((MATRIX_SIZE, MATRIX_SIZE))

mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(2, 2), ('x', 'y'))

mesh1D = jax.sharding.Mesh(jax.devices(), ('x'))

sharded_sharding_1D = jax.sharding.NamedSharding(mesh1D, jax.sharding.PartitionSpec('x'))
sharded_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x', 'y'))
unsharded_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))

A_sharded_sharding = jax.device_put(A, sharded_sharding)
A_unsharded_sharding = jax.device_put(A, unsharded_sharding)
A1D_sharded_sharding = jax.device_put(A, sharded_sharding_1D)

print('Visualize A_sharded_sharding')
jax.debug.visualize_array_sharding(A_sharded_sharding)

print('Visualize A_unsharded_sharding')
jax.debug.visualize_array_sharding(A_unsharded_sharding)

print('Visualize A1D_sharded_sharding')
jax.debug.visualize_array_sharding(A1D_sharded_sharding)

@partial(jax.jit, out_shardings=unsharded_sharding)
def unshard_array(input):
    return input

@partial(jax.jit, out_shardings=sharded_sharding)
def shard_array(input):
    return input


def print_z(z):
    print(f'{z=}')
    return z

@print_z
def print_a(x, y):
    print(f'{x=}, {y=}')
    return y

breakpoint()

