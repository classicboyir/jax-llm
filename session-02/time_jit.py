

import datetime
import jax
from jax import jit
from timing_util import simple_timeit

MATRIX_DIM = 32768
STEPS = 10

A = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM))
B = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM))
C = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM))

num_bytes = A.size * 4 # 4 as it's fp32
print('A.size -> ', A.size) # 1073741824
total_num_bytes_crossing_to_hbm = num_bytes * 3 # A, B, and C

total_num_flops = MATRIX_DIM * MATRIX_DIM


def f(A, B):
    return A + B

jit_f = jit(f)

average_time = simple_timeit(jit_f, A, B, task="mytask")/1000

print(f"{average_time}, tera flops per sec {total_num_flops/average_time / 10**12}, giga bytes per second {total_num_bytes_crossing_to_hbm/average_time / 10**9}")

## 
