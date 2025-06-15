# Graph Spring Neural ODE

Official jax implementation of the paper "[Graph Spring Neural ODEs for Link Sign Prediction](https://link.springer.com/article/10.1007/s10994-025-06794-1)". 

To cite this work, please use the following BibTeX entry:

```bibtex
@article{rehmann2025graph,
  title={Graph spring neural ODEs for link sign prediction},
  author={Rehmann, Andrin and Bovet, Alexandre},
  journal={Machine Learning},
  volume={114},
  number={7},
  pages={1--19},
  year={2025},
  publisher={Springer}
}
```

## Running the code

### Dependencies

Installation instructions for Ubuntu (conda venv recommended):

1. Install jax: 
``pip install -U "jax[cuda12]"``
2. Install pytorch (CPU version !) ``pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu``
3. Install other dependencies: ``pip install torch_geometric matplotlib scikit-learn pyyaml tqdm optax inquirer pandas``

Note that this repository relies on torch_geometric for dataloading capabilities, the Neural Network / Graph Neural Network implementation is however done from scratch in pure JAX. This is the reason this repo does not have a dependency on any JAX neural network libraries.
### Training

To start the training run ```python src/train.py <Dataset Name> params/train_params.yaml``` where ```<Dataset Name>``` can be either:

- BitcoinAlpha
- BitcoinOTC
- Epinions
- Slashdot 
- WikiRFA
- Tribes

the parameter file found at ```params/train_params.yaml``` can be adjusted as desired. The script will automatically download the datasets and cache them for later usage. This process can take a few seconds when executed for the first time.

A training run outputs a model file, which can be found under ```model/<ModelName>.yaml``` and a csv with the stats of the training under ```plots/data/training_process.csv```. The stats can be ploted by running ```python plots/forward.py``` which produces the following image:

![Forward](plots/forward.png)

### Testing

To test the model against different datasets execute ```ipython src/get_benchmarks.py <Dataset Name> params/test_params.yaml```. Note that we use IPython here (install with ```pip install ipython```) for accurate time measurements. The script will generate a .csv file in the format ```<Dataset Name>_<num dimensions>_<nn / spring>.csv```. The training times are displayed in the command line and of now, manually collected in the ```plots/data/speedup.csv``` file. You can generate a plot of speedups with  ```python plots/speedup.py``` such as:

![Forward](plots/performance.png)

## Important Code / Paper sections

The graph spring network layer as described in the paper is

$$
    \text{gsn}(\mathbf{x}_i) = g(\mathbf{y}_i) \cdot \sum_{j \in N_i} f \bigl(\mathbf{z}_{i,j}, d(\mathbf{x}_{i}, \mathbf{x}_{j})\bigr) \cdot \frac{\mathbf{x}_j - \mathbf{x}_i}{d(\mathbf{x}_i, \mathbf{x}_j)}.
$$

The spring network layer for the specific implementations SPR and SPR-NN can be found in ``src/euler.py``. In the same file, ``euler_step()`` implements the update method $\Phi$ as described in the paper as

$$
    \Phi \bigl( \mathbf{S}(t) \bigr) = \begin{bmatrix}
        1 & 0 \\ 0 & (1-d) 
    \end{bmatrix}\mathbf{S}(t) + dt \begin{bmatrix} \mathbf{V}(t) \\  \mathbf{F}(t)\end{bmatrix}.
$$

The whole forward simulation as described in the paper as 

$$
    \mathbf{S}(n ) = (\overbrace{\Phi \circ \dots \circ \Phi}^{n \text{-times}})(\mathbf{S}(0) )
$$

is implemented in ``src/simulation`` in the function ``simulate()``.

The loss function 
$$
    L(G, \textbf{S}(t)) = \sum_{(u, v) \in E} (\sigma(u, v) - {\sigma}(u, v))^2 \cdot \omega(u, v)
$$ 
can be found in ``src/simulation`` as ``loss()``, which is then combined with a forward simulation in ``simulate_and_loss()``. 

The gradient 
$$
    \nabla L  = \frac{\partial L}{\partial \mathbf{S}(n)} \frac{\partial \mathbf{S}(n)}{\partial \mathbf{p}} + \frac{\partial L}{\partial \mathbf{p}}
$$

is evaluated in ``src/trainer.py`` using the line 

```
value_and_grad_fn = jax.value_and_grad(sm.simulate_and_loss, argnums=3, has_aux=True)
```

### Graph preprocessing

All releveant code to graph preprocessing can be found in ``src/graph/graph.py`` and ``src/graph/split.py``.