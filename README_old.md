# Force Directed Node Embedding for Link Sign Prediction

This is the code for the paper called (Papername) which was published in (Journalname). The paper can be found here: (Link to paper)

## Running the code

### Dependencies

Installation instructions for Ubuntu (conda venv recommended):

1. Install jax: 
``pip install -U "jax[cuda12]"``
2. Install pytorch (CPU version !) ``pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu``
3. Install other dependencies: ``pip install torch_geometric matplotlib scikit-learn pyyaml tqdm optax inquirer pandas``

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

### Stable training

Sometimes the gradient will have NaN values, I am not entirely sure why this happens. However this can be fixed by ensuring that ```jax.config.update("jax_enable_x64", True)``` is set in the main script.

The training schedule is set with the following parameters:

```python
N_EPOCH_SEQUENCE = [50, 40, 30, 20, 20, 20, 40]
TRAIN_ITER_SEQUENCE = [50, 60, 70, 80, 100, 150, 200]
DT_SEQUENCE = [0.01, 0.01, 0.005, 0.005, 0.005, 0.002, 0.002]
DAMPING_SEQUENCE = [0.2, 0.1, 0.05, 0.05, 0.03, 0.03, 0.03]
```

This means that the first 50 iterations are trained with a time step of 0.01 and a damping factor of 0.2. Gradually the time step is decreased and the damping factor is increased. This is done to ensure that the loss surface is smooth and that the model is less likely to get stuck in a local minimum. The above parameters are rather arbitrary and there is probably a better way to set them.

Depending on the dataset and the number of simulation iterations during training, the required memory exceeds the available GPU memory (if JAX is installed in GPU mode). The required memory increases with the number of iterations, as the loss needs to be backpropagated through all iterations. In this case the graph can be partitioned into smaller subgraphs, the loss can be computed for each subgraph and the gradients can be applied after every batch. If desired one can also use gradient accumulation to further reduce the memory requirements by setting ``MULTISTPES_GRADIENT`` to a number larger than 1.

Enabling this feature can be done by setting ``GRAPH_PARTITIONING`` to ``True`` and the ``BATCH_NUMBER`` should be set to appropriate value. The batch number should be as small as possible but large enough to avoid the OMM error.

### Checkpoints

After a successful training the model parameters can optionally be saved. The model parameters are saved in the ``checkpoints`` folder and automatically loaded the next time you run the script. 