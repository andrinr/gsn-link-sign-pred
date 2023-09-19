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


## Method

### Task Definition

We are given a signed network $G = (V, E, \sigma)$, where $V$ is the set of nodes, $E$ is the set of edges and $\sigma: E \rightarrow \{-1, 1\}$ is the sign function. The task is to predict the sign of a given edge $(u, v) \in E$.

To do so we randomly sample a set of edges $E_{test} \subset E$ and set the sign to a neutral value 0. We want to find a function $f: V \times V \rightarrow \mathbb{R}$, which maps an edge $(u, v) \in E_{test}$ to a sign $\sigma(u, v) \in \{-1, 1\}$.

### Node Embedding

We compute a node embedding $x_u \in \mathbb{R}^d$ for each node $u \in V$. The embedding is computed using a spring network simulation, which is described in the next section. The embedding is then used to predict the sign of an edge $(u, v) \in E_{test}$ using a logistic regression model, which simply takes the distance between the two embeddings $x_u$ and $x_v$ as input.


### Spring Network Simulation

The node positions in this spring network are considered as node feature vectors. For all positive, neutral and negative edges fixed resting lengths $l^{+}$, $l^{0}$ and $l^{-}$ are defined. The actual length of an edge is denoted as $ L_i = \|{X_l - X_k}\|_2 $. 

The force acting on a node $(v_i)$ from the edge $(v_i, v_j)$ is the partial derivative of the energy with respect to the node position $\frac{\partial E(X_1, X_2) }{\partial X_1} $. We denote the force coming from negative, neutral and positive nodes as $ f^{-}_{i,j} $, $f^{0}_{i,j} $ and $ f^{-}_{i,j}$. The partial differential equations evaluate to equations:

$ f^{-}_{i,j} = \alpha^{-} \times min(l^{-} - L_i, 0) \frac{X_j - X_i}{\|{ X_j - X_i}\|_2} $

$ f^{0}_{i,j} = \alpha^{0} \times (l^{0} - L_i, 0) \frac{X_j - X_i}{\|{ X_j - X_i}\|_2} $

$ f^{+}_{i,j} = \alpha^{+} \times max(l^{+} - L_i, 0) \frac{X_j - X_i}{\|{ X_j - X_i}\|_2} $

where $L_i$ is the number of negative edges connected to node $v_i$ and $l^{-}$, $l^{0}$ and $l^{+}$ are the thresholds for the negative, neutral and positive force respectively. $\alpha^{-}$, $\alpha^{0}$ and $\alpha^{+}$ are the scaling factors for the negative, neutral and positive force respectively.

### 