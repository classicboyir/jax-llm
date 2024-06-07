import numpy as np
import jax

A = jax.numpy.ones((1024, 1024))

mesh1 = jax.sharding.Mesh(np.reshape(jax.devices(), (2,2)), ["myaxis1", "myaxis2"])
p = jax.sharding.PartitionSpec("myaxis1", "myaxis2")
sharding1 = jax.sharding.NamedSharding(mesh1, p)
sharded_A = jax.device_put(A, sharding1)

mesh2 = jax.sharding.Mesh(np.reshape(jax.devices(), (1,4)), ["myaxis1", "myaxis2"])
p = jax.sharding.PartitionSpec("myaxis1", "myaxis2")
sharding2 = jax.sharding.NamedSharding(mesh2, p)
sharded_B = jax.device_put(A, sharding2)

outcome = sharded_A + sharded_B

print('\n=========== sharded_A =========\n')
jax.debug.visualize_array_sharding(sharded_A)
print('\n=========== sharded_B =========\n')
jax.debug.visualize_array_sharding(sharded_B)

print('\n====================\n')
print('\n=========== Outcome =========\n')

jax.debug.visualize_array_sharding(outcome)



