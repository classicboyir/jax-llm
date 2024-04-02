import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn

BATCH_IN_SEQUENCE = 384
SEQUENCE_LENGTH = 128
VOCAB_DIM = 256
EMBED_DIM = 512


class TinyLLM(nn.Module):
    @nn.compact
    def __call__(self, x):
        '''
            x is [BATCH, SEQUENCE]
        '''

        embedding = self.param(
            'embedding',
            nn.initializers.normal(1),
            (VOCAB_DIM, EMBED_DIM),
            jnp.float32
            )
        print(f'type of embedding {type(embedding)}')
        print(f'embedding.shape -> {embedding.shape}')
        x = jnp.asarray(embedding)[x] # BATCH, SEQUENCE, EMD
        print(f'x.shape - {x.shape}')
        return x



def convert_to_ascii(string_array, max_length):
    print(f'string_array: ${string_array}')
    result = np.zeros((len(string_array), max_length), dtype=np.uint8)
    for i, string in enumerate(string_array):
        for j, char in enumerate(string):
            # ƒprint(f'char: {char}, type {type(char)}')
            if j >= max_length:
                break
            # print(f'char: {char}')
            result[i, j]= char
    return result


def input_to_output(np_array):
    zero_array = np.zeros((BATCH_IN_SEQUENCE, SEQUENCE_LENGTH), dtype = np.uint8)
    zero_array[:, 1:SEQUENCE_LENGTH] = np_array[:, 0:SEQUENCE_LENGTH - 1]
    return zero_array

def main():
    ds = tfds.load('lm1b', split='train', shuffle_files=False)
    ds = ds.batch(BATCH_IN_SEQUENCE)

    rngkey = jax.random.key(0)
    model = TinyLLM()
    params = model.init(rngkey, jnp.ones((BATCH_IN_SEQUENCE, SEQUENCE_LENGTH), dtype=jnp.uint8))
    
    for example in ds:
        print(example['text'])
        outputs = convert_to_ascii(example['text'].numpy(), SEQUENCE_LENGTH)
        inputs = input_to_output(outputs)
        model.apply(params, inputs)

        proposed_outputs = model.apply(params, inputs)

        breakpoint()
        break

if __name__ == "__main__":
    main()
    print("hello")