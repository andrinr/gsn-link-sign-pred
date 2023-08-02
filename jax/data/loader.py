import os
import requests
from abc import ABC, abstractmethod
from typing import Optional
import jax.numpy as jnp
from graph import Graph
class Loader(ABC):

    def __init__(
            self, 
            url: str,
            destination: str) -> None:
        
        self.url = url
        self.destination = destination

    def __call__(self) -> Graph:
        filename = self.url.split('/')[-1]
        filepath = os.path.join(self.destination, self.filename)
      
        if not os.path.exists(self.destination):
            os.makedirs(self.destination)

        if not os.path.exists(filepath):
            self.download(self.url, filepath)

        raw_data = jnp.load(filepath)
        return self.process(raw_data)
        
    @abstractmethod
    def process(self, 
        raw_data : jnp.ndarray) -> Graph:
        pass

    def download(
            self,
            url: str,
            filepath: str) -> None:

        req = requests.get(self.url, stream=True)

        if req.ok:
            with open(filepath, 'wb') as f:
                for chunk in req.iter_content(chunk_size=1024):
                    f.write(chunk)

        else:
            req.raise_for_status()