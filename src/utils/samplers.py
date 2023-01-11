import numpy as np

def even_uniform(min : int, max : int, size : int):
    """
    creates an integer array with an even sum of size `size` with values sampled from a uniform distribution

    args:
        min: the minimum value of the uniform distribution
        max: the maximum value of the uniform distribution
        size: the size of the array

    returns:
        an array of integers with an even sum
    """
    sample = np.random.randint(min, max, size)
    while np.sum(sample) % 2 == 1:
        sample = np.random.randint(min, max, size)
    return sample

def even_exponential(size : int, scale : float = 5.0):
    """ 
    creates an integer array with an even sum of size `size` with values sampled from an exponential distribution

    args:
        scale: the scale parameter of the exponential distribution
        size: the size of the array

    returns:
        an array of integers with an even sum
    """
    sample = np.random.exponential(scale, size).astype(int)
    while np.sum(sample) % 2 == 1:
        sample = np.random.exponential(scale, size).astype(int)
    return sample