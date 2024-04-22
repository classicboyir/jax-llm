

import datetime
import jax
from jax import jit
from timing_util import simple_timeit
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_2d_tuples(data, file_path):
    # Extract x and y coordinates from the data array of tuples
    x = [point[0] for point in data]
    y = [point[1] for point in data]
    texts = [point[2] for point in data]
    
    # Create a scatter plot
    plt.scatter(x, y)

    for i, text in enumerate(texts):
        plt.text(x[i], y[i], text, fontsize=9, ha='center', va='bottom')
   
    
    # Set title and labels
    plt.title('Roofline of MatMul for different Matrix Sizes')
    plt.xlabel('Opeartion Intensity')
    plt.ylabel('Performance (GFLOPS)')
    
    # Show the plot
    plt.savefig(file_path)
    plt.clf()


# MATRIX_DIM = 32768
rooflit_data = []
for MATRIX_DIM in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]:
    STEPS = 10

    NUM_MATRICES = 2**28 // MATRIX_DIM ** 2

    A = jax.numpy.ones((NUM_MATRICES, MATRIX_DIM, MATRIX_DIM), dtype=jnp.bfloat16)
    B = jax.numpy.ones((NUM_MATRICES, MATRIX_DIM, MATRIX_DIM), dtype=jnp.bfloat16)

    num_bytes = A.size * 2 # 4 as it's bfloat16
    # print('A.size -> ', A.size) # 1073741824
    total_num_bytes_crossing_to_hbm = num_bytes * 3

    total_num_flops = NUM_MATRICES * (2 * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM + MATRIX_DIM * MATRIX_DIM) # first three for matmul and the last two for relu

    print(f'{MATRIX_DIM=}')
    print(f'arithmetic intensity: {total_num_flops/total_num_bytes_crossing_to_hbm}')

    @jax.jit
    def f(A, B):
        return jax.lax.batch_matmul(A, B)

    average_time = simple_timeit(f, A, B, task="jit_it")/1000
    print(f"{average_time}, tera flops per sec {total_num_flops/average_time / 10**12}, giga bytes per second {total_num_bytes_crossing_to_hbm/average_time / 10**9}")
    print('\n\n\n')
    rooflit_data.append((total_num_flops/total_num_bytes_crossing_to_hbm, total_num_flops/average_time / 10**12, str(MATRIX_DIM)))
 
plot_2d_tuples(rooflit_data, 'roofline')
