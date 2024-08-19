from jax import config
from jax import numpy as jnp
from pathlib import Path

# Use this to change the default behavior
# float64 should have better convergence properties
# at the risk of being more memory storage
config.update("jax_enable_x64", True)
jnp_float = jnp.float64

_parent_dir = Path(__file__).parents[1]