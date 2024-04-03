import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state

import optax


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
        # print(f'type of embedding {type(embedding)}')
        # print(f'embedding.shape -> {embedding.shape}')
        x = embedding[x] # BATCH, SEQUENCE, EMD
        # print(f'x.shape - {x.shape}')

        for i in range(LAYERS):
            feedforward = self.param(
                'feedforward_' + str(i),
                nn.initializers.lecun_normal(),
                (EMBED_DIM, FF_DIM),
                jnp.float32
            )

            x = x @ feedforward
            x = jax.nn.relu(x)

            embed = self.param(
                'embed_' + str(i),
                nn.initializers.lecun_normal(),
                (FF_DIM, EMBED_DIM),
                jnp.float32
            )

            x = x @ embed
            x = jax.nn.relu(x)

        return x @ embedding.T # to rotate this to EMBED_DIM, VOCAB_DIM


def calculate_loss(params, model, inputs, outputs):
    one_hot = jax.nn.one_hot(outputs, VOCAB_DIM) # -> (384, 128, 256)
    proposed_outputs = model.apply(params, inputs) # -> (384, 128, 256)

    loss = optax.softmax_cross_entropy(proposed_outputs, one_hot)
    # breakpoint()
    return jnp.mean(loss)


def convert_to_ascii(string_array, max_length):
    # print(f'string_array: ${string_array}')
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
    _params = model.init(rngkey, jnp.ones((BATCH_IN_SEQUENCE, SEQUENCE_LENGTH), dtype=jnp.uint8))
    
    tx = optax.adam(learning_rate = LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params = _params,
        tx=tx
    )

    iter = 0
    for example in ds:
        # print(example['text'])
        outputs = convert_to_ascii(example['text'].numpy(), SEQUENCE_LENGTH)
        inputs = input_to_output(outputs)

        loss, grad = jax.value_and_grad(calculate_loss)(state.params, model, inputs, outputs)
        state = state.apply_gradients(grads = grad)
        print(f"{iter} -> loss:[{loss}] - tx:[]")
        iter += 1

        breakpoint()
        ## break

    breakpoint()

if __name__ == "__main__":
    main()
    print("hello")
