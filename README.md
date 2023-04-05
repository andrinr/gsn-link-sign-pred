# *SpringE*: Fast Node Representation Generation for Link Sign Prediction

A signed network is a graph where each edge represent either a positive or negative relationship between its two nodes. This paper proposes \textit{SpringE}, which is able to capture crucial information about the local node environment using low dimensional feature vectors. Contrary to other methods, no neural network is used, which greatly increases the runtime of the given method. 

## Requirements

You can try to install to setup the environment using the ```environment.yaml``` file. To do so run ```conda env create -f environment.yaml```. ( You need to have conda or conda-mini installed.)

If that does not work, you can install the following packages manually:

- tqdm
- numpy
- scipy
- pyg-nightly (Many unreleased features are used, so you need to install the nightly version)
- skikit-learn
- torch
- pyyaml
- pandas
- matplotlib
- nevergrad
- networkx (Install direclty from source as unreleased features are used.)

In order to use the GPU, you need to install CUDA and PyTorch with CUDA support.

## Running the code

Run ```python src/main.py``` where the following arguments can opionally be passed to the script. 

- ```-i <value>``` sets the number of iterations for the spring network simulation. Default is 1000.
- ```-d <value>``` sets the damping factor for the spring network simulation. Default is  0.02.
- ```-h <value>``` sets the size of a timestep for the spring network simulation. Default is 0.005.
- ```-s <value>``` sets the dimensionality of the node embeddings. Default is 64.
- ```-o <value>``` if set starts the nevergrad optimization with the given number of iterations. The recommendations can then be stored as the new default parameters in the ```params.yaml``` file. Recommended is around 30 iterations.

The script asks the user to select a dataset in the terminal after execution.

## Testing the code

run ```pytest``` in the ```src``` folder.


