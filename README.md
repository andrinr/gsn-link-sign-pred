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

We are given a signed network $G = (V, E, \sigma)$, where $V$ is the set of nodes and $E$ is the set of edges. Finally for each edge $(i, j)$ there is a sign which denotes a positive or a negative relationship between the two nodes. Formally we introduce a sign function $\sigma: V \times V \rightarrow \{-1, 0, 1\}$, which maps an edge $(i, j) \in E$ to a sign $\sigma(i, j) \in \{-1, 0, 1\}$.

The task is to predict the sign of an edge $(u, v) \in E_{test}$, where $E_{test} \subset E$ is a set of edges which are not known to the algorithm.

To do so we randomly sample a set of edges $E_{test} \subset E$ and set the sign to a neutral value 0. We want to find a function $f: V \times V \rightarrow \mathbb{R}$, which maps an edge $(u, v) \in E_{test}$ to a sign $\sigma(u, v) \in \{-1, 1\}$.

### Node Embedding

We compute a node embedding $x_u \in \mathbb{R}^d$ for each node $u \in V$. The embedding is computed using a phyiscal spring simulation, where each edge is modeled as a spring. The springs exert a force on the attached nodes. The nodes are then moved according to the forces acting on them. After a fixed number of iterations, the node positions can be used as node embeddings. To predict the sign of an edge $(u, v) \in E_{test}$, we simply take the distance between the two embeddings $x_u$ and $x_v$ and compare it to a threshold. If the distance is below the threshold, we predict a positive sign, otherwise we predict a negative sign.

The intuition behind this is, that nodes which are connected by a positive edge should be close to each other, while nodes which are connected by a negative edge should be far away from each other.

The plotted node embeddiings for the tribes dataset are shown below:

![Results](img/tribes_embeddings.png)

You can observe that all the tribes are grouped in 3 clusters. The tribes in the same cluster are connected only connected by positive edges, while the tribes in different clusters are connected by negative edges. The tribes dataset is a very simple dataset, and can be solved with only 2 dimensional node embeddings with a 100% accuracy using a 80% train and 20% test split. However for more complex dataset the number of dimensions needs to be increased to encode more complex relationships.

### Spring Network Simulation

We denote $x_i$ as the position of a node $i$. In this context the term position and embedding are used interchangably, as in the end of the algorithm, the position is direclty used as an embedding. Furthermore we use $N_i$ to denote the set of neighbors of node $i$ and $N^{+}_i$ to denote the set of neighbors of node $i$ which are connected by a positive edge. The sets $N^{-}_i$ and $N^{0}_i$ are defined analogously. Furthermore we store a velocity $v_i$ for each node $i$.

For all positive, neutral and negative edges fixed resting lengths $l^{+}$, $l^{0}$ and $l^{-}$ are defined. Furthermore each edge type has a stiffness $\alpha^{+}$, $\alpha^{0}$ and $\alpha^{-}$.

The force acting on a node $(v_i)$ from the edge $(v_i, v_j)$ is the partial derivative of the energy with respect to the node position $\frac{\partial E(x_i, x_j) }{\partial x_i} $. We denote the force coming from negative, neutral and positive nodes as $f^{-}_{i,j}$, $f^{0}_{i,j}$ and $f^{-}_{i,j}$. The partial differential equations evaluate to equations:

$$f^{-}_{i,j} = \alpha^{-} \cdot min(l^{-} - \|{ X_j - X_i}\|_2, 0) \frac{X_j - X_i}{\|{ X_j - X_i}\|_2}$$

$$f^{0}_{i,j} = \alpha^{0} \cdot (l^{0} - \|{ X_j - X_i}\|_2, 0) \frac{X_j - X_i}{\|{ X_j - X_i}\|_2}$$

$$f^{+}_{i,j} = \alpha^{+} \cdot max(l^{+} - \|{ X_j - X_i}\|_2, 0) \frac{X_j - X_i}{\|{ X_j - X_i}\|_2}$$

Therefore the total force acting on a node $v_i$ is:

$$f_i = \sum_{j \in N^-(i)} f^{-}_{i,j} + \sum_{j \in N^0(i)} f^{0}_{i,j} + \sum_{j \in N^+(i)} f^{+}_{i,j}$$

The total force acting on a node $v_i$ is then used to update the position of the node $v_i$ using the kick drift kick method. We use the notation $x_i(t)$ to denote the position of node $v_i$ at time $t$ and $v_i(t)$ to denote the velocity of node $v_i$ at time $t$. To avoid divergence of the simulation, we use a viscous damping factor $d$.

$$v_i(t + \frac{h}{2}) = v_i(t) + \frac{h}{2} \cdot f_i(t) - d \cdot \|{v_i(t)}\|_2$$

$$x_i(t + h) = x_i(t) + h \cdot v_i(t + \frac{h}{2})$$

$$v_i(t + h) = v_i(t + \frac{h}{2}) + \frac{h}{2} \cdot f_i(t + h) - d \cdot \|{v_i(t + \frac{h}{2})}\|_2$$

where $h$ is the size of a timestep.

The simulation is run for a fixed number of iterations. After the simulation has converged, the node positions are used as node embeddings. To predict a sign of an edge $(u, v) \in E_{test}$, we simply take the distance between the two embeddings $x_u$ and $x_v$ and compare it to a threshold. If the distance is below the threshold, we predict a positive sign, otherwise we predict a negative sign.

Now how do we find the optimal parameters for the simulation? Either we use educated guesses, but this is not very efficient. 

### Differentiable Simulation for Parameter Optimization

Using automatic differentiation we are able to compute the derivative of the simulation with respect any input parameter. This allows us to optimize the parameters of the simulation using gradient descent. We use the following loss function to optimize the parameters of the simulation:

For a predefined distance threshold $d_{th}$, the predicted sign of an edge $(u, v)$ is computed using a sigmoid function:

$$\sigma^{\prime}(u, v) = \frac{1}{1 + e^{ \|{x_u - x_v}\|_2 - d_{th}}}$$

We then compute the loss function as follows:

$$l = \frac{1}{|E|} \sum_{(u, v) \in E} (\sigma^{\prime}(u, v) - \sigma(u, v))^2 \cdot \omega(u, v)$$

where $n$ is a normalization function which for a positive edge returns one divided by the number of positive edges, for a neutral edge one divided by the number of neutral edges and for a negative edge one divided by the number of negative edges.

$$\omega(u, v) = \begin{cases} 
\frac{1}{|E^{+}|} & \text{if } \sigma(u, v) = 1 \\ 
\frac{1}{|E^{-}|} & \text{if } \sigma(u, v) = -1 
\end{cases}$$


We used JAX for every computation in the function, this allows use to take the simulation derivative with respect any specified parameter. This yields a gradient which can be used to optimize the parameters of the simulation. Initially the loss was applied to the spring parameters $d_{th}$, $l^{+}$, $l^{0}$, $l^{-}$, $\alpha^{+}$, $\alpha^{0}$ and $\alpha^{-}$ with the following results:

| Parameter | Value | Description |
| --- | --- | --- |
| $d_{th}$ | 14.420479774475098 | Distance threshold for the sigmoid function. |
| $l^{+}$ | 9.82265567779541 | Resting length of a positive edge. |
| $l^{0}$ | 16.775850296020508 | Resting length of a neutral edge. |
| $l^{-}$ | 20.94917106628418 | Resting length of a negative edge. |
| $\alpha^{+}$ | 7.730042457580566 | Stiffness of a positive edge. |
| $\alpha^{0}$ | 1.1881768703460693 | Stiffness of a neutral edge. |
| $\alpha^{-}$ | 14.80991268157959 | Stiffness of a negative edge. |

We can see how the network learns, with a 16 dimensional embedding on the Bitcoin Alpha dataset:

![Results](img/loss_spring_params_16.png)

The main reason why the loss and the measures are so 'jagged' is that the initial condiations of the node positions are randomized for each epoch. 

![Results](img/measures_spring_params_16.png)

![Results](img/spring_params_16.png)

### Message Passing Neural Network for local pattern recognition

As of now we have hardcoded the forces acting on a node as:

$$f_i = \sum_{j \in N^-(i)} f^{-}_{i,j} + \sum_{j \in N^0(i)} f^{0}_{i,j} + \sum_{j \in N^+(i)} f^{+}_{i,j}$$ 

The above assumption is based on the social balance theory and does apply to many dynamics in networks. (Add some sort of evidence?)
However there might be more complex patterns in the network, which we are not able to capture with the above method. Therefore we are looking for a function $B$, which given an edge $(u, v)$ and two auxillary information vectors $m_u$ and $m_v$ for the nodes $u$ and $v$ respectively, computes values $s^{-}$, $s^{0}$ and $s^{+}$ which denote the strength of the positive and negative relationship between the two nodes. We then compute the forces acting on the nodes as follows:

$$f_i = \sum_{j \in N} f^{-}_{i,j} \cdot s^{-} + f^{+}_{i,j} \cdot s^{+} + f^{0}_{i,j} \cdot s^{0}$$

The auxillary vectors $m_i \in \mathbb{R}^d$ are computed using a message passing neural network $A$. 

#### Message Passing Neural Network $A$

A message passing neural network is a neural network which is applied for each node on all of its input edges. This neural network is usually applied for several recursive iterations, which leads to information propagating thrue the graph. The auxillary information vectors $m_i$ are initialized as random vectors. 

A single step of a message passing neural network on a node $i$ is defined as follows:

$m_i(t+1) = \Phi \left[ m_i(t), \mathop{\bigoplus}\limits_{j \in N_i} \Psi \left( m_i(t), m_j(t), \sigma(i, j) \right) \right]$

where $\Phi$ and $\Psi$ are neural networks and $\oplus$ is a permutation invariant aggregation function. Examples for aggregation functions are sum, mean, max, min, etc.

#### Force decision network $B$

Network $B$ maps between two nodes $u$ and $v$ and their auxillary information vectors $m_u$ and $m_v$ to a vector $s \in \mathbb{R}^3$, which denotes the strength of the positive and negative relationship between the two nodes. The network is defined as follows:

$$s = B(m_u, m_v, \sigma(u, v))$$

#### Training Step

A trainig step of the entire process then looks as follows:

1. Sample $x_i$ and $m_i$ for all nodes $i \in V$ from a normal distribution. Initialize $v_i$ to zero.
2. Run the message passing neural network $A$ for a fixed number of iterations.
3. Run the force decision network $B$ for all edges $(u, v) \in E$. 
4. Run the spring network simulation for a fixed number of iterations.
5. Predict the sign of an edge $(u, v)$ using the distance between $x_u$ and $x_v$.
6. Compute the loss and update the parameters of the message passing neural network $A$ and the force decision network $B$ using gradient descent.
7. Repeat from step 1.

## Training

The training process of the neural networks is challenging as a single simulation run takes up to a minute on my machine on all of the datasets. Therefore we use the Tribes dataset as a pretraining and then finetune the network on the other datasets. This saves a lot of time as the Tribes dataset is very small and can be simulated very quickly.

## Results

### Comparison to other previous best method

![Results](results.png)

### Energy minima and score correlations

Generally a decrease in energy in the system correlates with a better performance of the method. However the effect is much more pronounced in the beginning of the method.

![Results](energy_score_corr.png)


### Correlation between energy 
![Results](ratio_energy.png)


## Sources:

For aggregation methods and graph subsampling:
@book{hamilton_ying_leskovec, title={Inductive Representation Learning on Large Graphs}, url={https://arxiv.org/pdf/1706.02216.pdf}, author={Hamilton, William and Ying, Rex and Leskovec, Jure} }

‌