import numpy as np

def even_uniform(min : int, max : int, size : int):
    sample = np.random.randint(min, max, size)
    while np.sum(sample) % 2 == 1:
        sample = np.random.randint(min, max, size)
    return sample

def even_exponential(scale : float, size : int):
    """ 
    creates an integer array of size `size` with values sampled from an exponential distribution

    """
    sample = np.random.exponential(scale, size).astype(int)
    while np.sum(sample) % 2 == 1:
        sample = np.random.exponential(scale, size).astype(int)
    return sample