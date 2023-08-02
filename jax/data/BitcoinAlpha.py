import jax.numpy as jnp
from data import Loader
from graph import Graph

class BitcoinAlpha(Loader):

    def process(self, raw_data: jnp.ndarray) -> Graph:
        print(raw_data)
        return super().process(raw_data)