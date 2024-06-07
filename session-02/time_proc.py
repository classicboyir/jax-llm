

import datetime
import jax
import numpy as np

MATRIX_DIM = 65536 # 32768
STEPS = 10

A = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM))

num_bytes = A.size * 4 # 4 as it's fp32
print('A.size -> ', A.size) # 1073741824

B = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM))

# num_bytes = A.size * 4 # 4 as it's fp32
# print('A.size -> ', A.size) # 1073741824
total_num_bytes_crossing_to_hbm = num_bytes * 3 # A, B, and C

total_num_flops = MATRIX_DIM * MATRIX_DIM

jax.profiler.start_trace('gs://maxtext-logs-dogfood-proj/jax-llm/dev-env/raw_1') ## -> tensorboard --logdir=gs://maxtext-logs-dogfood-proj/jax-llm/dev-env/raw
starttime = datetime.datetime.now()

for i in range(STEPS):
    C = A + B

endtime = datetime.datetime.now()
jax.profiler.stop_trace()

average_time = (endtime - starttime).total_seconds() / STEPS

print(f"{average_time}, tera flops per sec {total_num_flops/average_time / 10**12}, giga bytes per second {total_num_bytes_crossing_to_hbm/average_time / 10**9}")
