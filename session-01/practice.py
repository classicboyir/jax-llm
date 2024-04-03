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
FF_DIM = 2048

LAYERS = 4

HEAD_DEPTH = 128
NUM_HEADS = 4

LEARNING_RATE = 1e-3

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

        for i in range(LAYERS):
            feedforward = self.param(
                'feedforward_' + str(i),
                nn.initializers.lecun_normal(),
                (EMBED_DIM, FF_DIM),
                jnp.float32
            )

            x = x @ jnp.asarray(feedforward)
            x = jax.nn.relu(x)

            embed = self.param(
                'embed_' + str(i),
                nn.initializers.lecun_normal(),
                (FF_DIM, EMBED_DIM),
                jnp.float32
            )

            x = x @ jnp.asarray(embed)
            x = jax.nn.relu(x)

        return x @ jnp.asarray(embedding).T # to rotate this to EMBED_DIM, VOCAB_DIM



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
        print(f'proposed_outputs.shape - {proposed_outputs.shape}')

        breakpoint()
        break

if __name__ == "__main__":
    main()
    print("hello")

params['params'].keys()
['embedding', 'feedforward_0', 'embed_0', 'feedforward_1', 'embed_1', 'feedforward_2', 'embed_2', 'feedforward_3', 'embed_3']