# SpringE: Fast Non-Neural Network Node Representation Generation for Link Sign Prediction

Tasks in network analysis often involve predicting on a node, edge or
graph level. Signed networks, such as social media networks, contain signs that indicate the nature of the relationship between two
associated nodes, such as trust or distrust, friendship or animos-
ity, or influence or opposition. In this paper, we propose SpringE, a
node representation learning algorithm for link sign prediction with
comparable performance as graph neural network based methods.
SpringE directly models the desired properties as an energy gradi-
ent using a physics-inspired spring network simulation based on as-
sumptions from structural balance theory which can be solved using
standard numerical integration methods for ODE.

## Requirements

- tqdm
- numpy
- scipy
- networkx
- pyg-nightly (Many unreleased features are used, so you need to install the nightly version)
- skikit-learn
- torch
- pyyaml
- pandas
- matplotlib

In order to use the GPU, you need to install CUDA and PyTorch with CUDA support.

## Running the code

Run ```python src/main.py``` where additional arguments can be passed to the script. 

- ```-i <value>``` sets the number of iterations for the spring network simulation
- ```-d <value>``` sets the damping factor for the spring network simulation
- ```-h <value>``` sets the size of a timestep for the spring network simulation
- ```-s <value>``` sets the dimensionality of the node embeddings
- ```-o <value>``` if set starts the nevergrad optimization with the given number of iterations. The recommendations can then be stored as the new default parameters in the ```params.yaml``` file.

The script asks the user to select a dataset in the terminal after execution.

