# Graph Spring Neural ODE

Official jax implementation of the paper "[Graph Spring Neural ODEs for Link Sign Prediction](https://arxiv.org/abs/2412.12916)". 

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
