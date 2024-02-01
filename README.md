# *SpringE*: Fast Node Representation Generation for Link Sign Prediction

This is the code for the paper called (Papername) which was published in (Journalname). The paper can be found here: (Link to paper)

## Table of Contents

- [Running the code](#running-the-code)
- [Introduction](#introduction)
- [JAX and Automatic Differentiation](#jax-and-automatic-differentiation)
- [Message Passing Networks](#message-passing-networks)
- [Training process](#training-process)


## Running the code

### Dependencies

Try ``conda create --name <env> --file env.yaml`` to create a conda environment with all the dependencies. Then activate the environment with ``conda activate <env>``.

If this does not work, try to install the dependencies manually. Make sure to use a Python version lower than the latest version as some of the dependencies are sometimes not compatible with the latest version (pytorch for example). 

Also make sure to install pytorch in CPU only mode and JAX with GPU support, this will make sure that JAX and pytorch do not conflict with each other.

### Running the code

Simply do ``python main.py`` to run the code. It will ask you to select a dataset which is then automatically downloaded and preprocessed. 

All important constants can be found on the very top of the main.py file. 

### Training the parameters

To train the model set ``OPTIMIZE_HEURISTIC_FORCE = True``. Depending on the dataset and the number of simulation iterations during training, the required memory exceeds the available GPU memory (if JAX is installed in GPU mode). The required memory increases with the number of iterations, as the loss needs to be backpropagated through all iterations.

In the case of a OMM error ``GRAPH_PARTITIONING`` can be set to ``True`` and the ``BATCH_NUMBER`` should be set to appropriate value. The batch number should be as small as possible but large enough to avoid the OMM error.

Once the training is finished, the script will ask you if you want to save the trained parameters. If you do, the parameters will be saved in the ``checkpoints`` folder and automatically loaded the next time you run the script.

## Introduction

## Jax and Automatic Differentiation

Most programms can be decomposed into a series of differentibale arithmetic operations. Using the chain rule, the partial derivative of very complex functions can be calculated even inlcuding control flow. This is called automatic differentiation. 

Automatic differentiation is a crucial part of the backpropagation algorithm used in neural networks. It computes the gradient of the loss function with respect to the parameters of the model. This same approach can be used to compute the gradient of any function with respect to any of its input parameters. Recently there has been a lot of research, where automatic differentiation is used to optimize functions other than neural networks.

JAX is a library for high-performance numerical computing that can automatically differentiate native Python and NumPy functions. It can differentiate through loops, branches, recursion, and closures, and it can take derivatives of derivatives of derivatives. It supports reverse-mode differentiation (a.k.a. backpropagation) via grad as well as forward-mode differentiation, and the two can be composed arbitrarily to any order. JAX also allows for JIT (Just in Time) compilation of functions. This can be used to speed up the execution of functions significantly.

Since there is no library for graph neural networks which works with a edge index representation, I have implemented the message passing network myself. The logic of it can be found in springs.py. For performance reasons it is crucial to refrain from using for loops and instead use matrix operations. 

For example to obtain the corresponding node positions of each edge we use:
    
```python
position_i = state.position[edge_index[0]]
position_j = state.position[edge_index[1]]
```

This can then be used to compute the accelerations for each edge. However we need to sum over all edge velocities which are connected to the same node. This can be achieved as follows:

```python
node_accelerations = jnp.zeros_like(spring_state.position)
node_accelerations = node_accelerations.at[edge_index[0]].add(edge_acceleration)
```