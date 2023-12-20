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