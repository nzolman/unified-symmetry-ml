from jax import random, jit
from jax import numpy as jnp
from jax.experimental.ode import odeint

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def get_mlp(nonlinearity = jnp.tanh):
    @jit
    def mlp(params, x):
        # per-example predictions
        activations = x
        for w, b in params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = nonlinearity(outputs)

        final_w, final_b = params[-1]
        logits = jnp.dot(final_w, activations) + final_b
        return logits
    return mlp