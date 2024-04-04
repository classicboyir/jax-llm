

import datetime
import jax
from timing_util import simple_timeit

MATRIX_DIM = 32768
STEPS = 10

A = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM))
B = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM))

num_bytes = A.size * 4 # 4 as it's fp32
print('A.size -> ', A.size) # 1073741824
total_num_bytes_crossing_to_hbm = num_bytes * 3 # A, B, and C

total_num_flops = MATRIX_DIM * MATRIX_DIM

def f(A, B):
    return A + B

jax.profiler.start_trace('/tmp/profile_me') ## -> tensorboard --logdir=/tmp/profile_me --load_fast=false --bind_all
starttime = datetime.datetime.now()

for i in range(STEPS):
    C = A + B

endtime = datetime.datetime.now()
jax.profiler.stop_trace()
average_time = (endtime - starttime).total_seconds() / STEPS

print(f"{average_time}, tera flops per sec {total_num_flops/average_time / 10**12}, giga bytes per second {total_num_bytes_crossing_to_hbm/average_time / 10**9}")
