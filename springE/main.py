# External dependencies
import sys
import inquirer
from torch_geometric.utils import is_undirected
from torch_geometric.loader import ClusterData, ClusterLoader
import yaml
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import torch_geometric.transforms as T
import numpy as np
import jax
from timeit import default_timer as timer

# Local dependencies
import simulation as sm
import graph as g
from io_helpers import get_dataset
import stats as stats

def main(argv) -> None:
    """
    Main function
    """

    EMBEDDING_DIM = 20

    # Paths
    DATA_PATH = 'springE/data/'
    CECKPOINT_PATH = 'training_checkpoints/'
    SCHEDULE_PATH = 'schedule/'

    # Define the range of the random distribution for the initial positions
    INIT_POS_RANGE = 1.0
    TEST_SHOTS = 5

    # TRAINING PARAMETERS
    MULTISTPES_GRADIENT = 1
    GRAPH_PARTITIONING = False

    # Deactivate preallocation of memory to avoid OOM errors
    #os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".95"
    #os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

    # Set the precision to 64 bit in case of NaN gradients
    jax.config.update("jax_enable_x64", True)

    questions = [inquirer.Checkbox('multiples',
        message="Select the options: (Press <space> to select, Enter when finished).",
        choices=['Use Neural Force (SE-NN)', 'Train Parameters', 'Convert to undirected', 'Partition Graph'],
        default=['Convert to undirected']),
    ]

    answers = inquirer.prompt(questions)
    
    # read answers from inquirer
    train_parameters = False
    use_neural_froce = False
    convert_to_undirected = False
    graph_partitioning = False
    number_of_subgraphs = 10
    if 'Train Parameters' in answers['multiples']:
        train_parameters = True
    if 'Use Neural Force (SE-NN)' in answers['multiples']:
        use_neural_froce = True
    if 'Convert to undirected' in answers['multiples']:
        convert_to_undirected = True
    if 'Partition Graph' in answers['multiples']:
        graph_partitioning = True

    if graph_partitioning:
        questions = [inquirer.Text('n_subgraphs',
            message="Enter the number of subgraphs",
            default='10'),
        ]
        answers = inquirer.prompt(questions)
        number_of_subgraphs = int(answers['n_subgraphs'])
    
    dataset, dataset_name = get_dataset(DATA_PATH, argv) 
    if not is_undirected(dataset.edge_index) and convert_to_undirected:
        transform = T.ToUndirected(reduce="min")
        dataset = transform(dataset)

    # uncomment to print dataset information
    # num_nodes = dataset.num_nodes
    # num_edges = dataset.num_edges
    # s = stats.Triplets(dataset)(n_triplets=3000, seed=0)
    # print(f"num_nodes: {num_nodes}")
    # print(f"num_edges: {num_edges}")
    # print(s.p_balanced)
    # print(dataset.edge_attr)
    # print(f"percentage of positive edges: {torch.sum(dataset.edge_attr == 1) / dataset.num_edges}")
        
    force_params_path = f"{CECKPOINT_PATH}{'neural' if use_neural_froce else 'spring'}_params_{EMBEDDING_DIM}.yaml"
    print(force_params_path)
    load_checkpoint = False
    if os.path.exists(force_params_path):
        questions = [
            inquirer.List('load',
                message=f"We found a checkpoint for {'neural' if use_neural_froce else 'spring'} force parameters. Do you want to load it?",
                choices=['Yes', 'No'],
            )]
        
        answers = inquirer.prompt(questions)
        if answers['load'] == 'Yes':
            load_checkpoint = True

    batches = []
    if GRAPH_PARTITIONING and train_parameters:
        cluster_data = ClusterData(
            dataset, 
            num_parts=number_of_subgraphs,
            keep_inter_cluster_edges=True)
        
        loader = ClusterLoader(cluster_data)
        for batch in loader:
            signedGraph = g.to_SignedGraph(batch, convert_to_undirected)
            print(f"num_nodes: {signedGraph.num_nodes}")
            if signedGraph.num_nodes > 50:
                batches.append(signedGraph)
    else:
        batches.append(g.to_SignedGraph(dataset, convert_to_undirected))
        
    if os.path.exists(force_params_path) and load_checkpoint:
        stream = open(force_params_path, 'r')
        force_params = yaml.load(stream,  Loader=yaml.UnsafeLoader)
    elif use_neural_froce:
        force_params = sm.NeuralForceParams(
            friend=sm.MLP(
                w0=jax.random.normal(random.PRNGKey(0), (7, 4)),
                b0=jnp.zeros(4),
                w1=jax.random.normal(random.PRNGKey(1), (4,1)),
                b1=jnp.zeros(1)),
            neutral=sm.MLP(
                w0=jax.random.normal(random.PRNGKey(2), (7, 4)),
                b0=jnp.zeros(4),
                w1=jax.random.normal(random.PRNGKey(3), (4,1)),
                b1=jnp.zeros(1)),
            enemy=sm.MLP(
                w0=jax.random.normal(random.PRNGKey(4), (7, 4)),
                b0=jnp.zeros(4),
                w1=jax.random.normal(random.PRNGKey(5), (4,1)),
                b1=jnp.zeros(1)),
        )
    else:
        force_params = sm.SpringForceParams(
            friend_distance=1.0,
            friend_stiffness=1.0,
            neutral_distance=1.0,
            neutral_stiffness=0.1,
            enemy_distance=5.5,
            enemy_stiffness=2.0,
            degree_multiplier=3.0)
            
    schedule_params_path = f"{SCHEDULE_PATH}{'neural' if use_neural_froce else 'spring'}_schedule.yaml"
    if os.path.exists(schedule_params_path):
        stream = open(schedule_params_path, 'r')
        schedule_params = yaml.load(stream,  Loader=yaml.UnsafeLoader)
    elif train_parameters:
        raise ValueError("no training schedule params found")

    # Create initial values for neural network parameters
    key_force, key_training, key_test = random.split(random.PRNGKey(2), 3)

    force_params = force_params

    if train_parameters:
        loss_hist = []
        metrics_hist = []
        force_params_hist = []
        dt_hist = []
        damping_hist = []
        iterations_hist = []

        for index, settings in enumerate(zip(
            schedule_params['number_of_simulations'],
            schedule_params['simulation_iterations'],
            schedule_params['simulation_dt'],
            schedule_params['simulation_damping'])):

            number_of_simulations = settings[0]
            iterations = settings[1]
            dt = settings[2]
            damping = settings[3]

            simulation_params_train = sm.SimulationParams(iterations=iterations, dt=dt, damping=damping)

            print(f"Training run {index + 1} of {len(schedule_params['number_of_simulations'])}")
            print(f"Running {number_of_simulations} simulations with {iterations} iterations each")
            print(f"Simulation parameters are dt: {dt}, damping: {damping}")

            force_params, loss_hist_, metrics_hist_, force_params_hist_ = sm.train(
                random_key=key_training,
                batches=batches,
                use_neural_force=use_neural_froce,
                force_params=force_params,
                training_params= sm.TrainingParams(
                    num_epochs=number_of_simulations,
                    learning_rate=schedule_params['learning_rate'],
                    batch_size=number_of_subgraphs,
                    init_pos_range=INIT_POS_RANGE,
                    embedding_dim=EMBEDDING_DIM,
                    multi_step=MULTISTPES_GRADIENT),
                simulation_params=simulation_params_train)
            
            #concatenate lists
            loss_hist = loss_hist + loss_hist_
            metrics_hist = metrics_hist + metrics_hist_
            force_params_hist = force_params_hist + force_params_hist_

            dt_hist = dt_hist + list(jnp.ones(number_of_simulations) * dt)
            iterations_hist = iterations_hist + list(jnp.ones(number_of_simulations) * iterations)
            damping_hist = damping_hist + list(jnp.ones(number_of_simulations) * damping)

        # plot loss over time
        epochs = np.arange(0, len(loss_hist))
        plt.plot(epochs, loss_hist)
        plt.title('Loss')
        plt.show()

        # get set1 color map
        colors = plt.cm.get_cmap('Set1', 10)

        # Three subplots with shared x axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        # plot metrics over time
        ax1.plot(epochs, [metrics.auc for metrics in metrics_hist], color=colors(0))
        ax1.plot(epochs, [metrics.f1_macro for metrics in metrics_hist], color=colors(1))
        ax1.legend(['AUC', 'F1 macro'], loc='lower left')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')

        ax2.plot(epochs, dt_hist, color=colors(2))
        ax2.plot(epochs, damping_hist, color=colors(3))

        ax2.legend(['dt', 'damping'], loc='lower left')
        ax2.set_xlabel('Epochs')

        ax3.plot(epochs, iterations_hist, color=colors(4))
        ax3.legend(['iterations'], loc='lower left')
        ax3.set_xlabel('Epochs')

        plt.show()

    shot_metrics = []
    times = []
    key_shots = random.split(key_test, TEST_SHOTS)     
    graph = {}

    for shot in range(TEST_SHOTS):

        graph = g.to_SignedGraph(dataset, convert_to_undirected)

        training_signs = graph.sign.copy()
        training_signs = jnp.where(graph.train_mask, training_signs, 0)
        training_graph = graph._replace(sign=training_signs)
        training_signs_one_hot = jax.nn.one_hot(training_signs + 1, 3)
        training_graph = training_graph._replace(sign_one_hot=training_signs_one_hot)

        start_time = timer()
        print(f"Running shot {shot + 1} of {TEST_SHOTS}")
   
        # initialize spring state
        spring_state = sm.init_spring_state(
            rng=key_shots[shot],
            n=graph.num_nodes,
            m=graph.num_edges,
            range=INIT_POS_RANGE,
            embedding_dim=EMBEDDING_DIM
        )

        simulation_params_test = sm.SimulationParams(
            iterations=schedule_params['test_iterations'],
            dt=schedule_params['test_dt'],
            damping=schedule_params['test_damping'])
        
        spring_state = sm.simulate(
            simulation_params=simulation_params_test,
            spring_state=spring_state, 
            use_neural_force=use_neural_froce,
            force_params=force_params,
            graph=training_graph)
        
        end_time = timer()
        print(f"Shot {shot + 1} took {end_time - start_time} seconds")
        times.append(end_time - start_time)


        metrics, _ = sm.evaluate(
            spring_state,
            graph.edge_index,
            graph.sign,
            graph.train_mask,
            graph.test_mask)
        
        shot_metrics.append(metrics)

    # print average metrics over all shots
    print(f"average metrics over {TEST_SHOTS} shots:")
    print(f"auc: {np.mean([metrics.auc for metrics in shot_metrics])}")
    print(f"f1_binary: {np.mean([metrics.f1_binary for metrics in shot_metrics])}")
    print(f"f1_micro: {np.mean([metrics.f1_micro for metrics in shot_metrics])}")
    print(f"f1_macro: {np.mean([metrics.f1_macro for metrics in shot_metrics])}")
    # print(f"true_positives: {np.mean([metrics.true_positives for metrics in shot_metrics])}")
    # print(f"false_positives: {np.mean([metrics.false_positives for metrics in shot_metrics])}")
    # print(f"true_negatives: {np.mean([metrics.true_negatives for metrics in shot_metrics])}")
    # print(f"false_negatives: {np.mean([metrics.false_negatives for metrics in shot_metrics])}")

    # standard deviation
    print(f"std auc: {np.std([metrics.auc for metrics in shot_metrics])}")
    print(f"std f1_binary: {np.std([metrics.f1_binary for metrics in shot_metrics])}")
    print(f"std f1_micro: {np.std([metrics.f1_micro for metrics in shot_metrics])}")
    print(f"std f1_macro: {np.std([metrics.f1_macro for metrics in shot_metrics])}")
    # print(f"std true_positives: {np.std([metrics.true_positives for metrics in shot_metrics])}")
    # print(f"std false_positives: {np.std([metrics.false_positives for metrics in shot_metrics])}")
    # print(f"std true_negatives: {np.std([metrics.true_negatives for metrics in shot_metrics])}")
    # print(f"std false_negatives: {np.std([metrics.false_negatives for metrics in shot_metrics])}")  

    print(f"average time per shot: {np.mean(times)}")

    if train_parameters:
        questions = [
            inquirer.List('save',
                message="Do you want to save the force parameters?",
                choices=['Yes', 'No'],
            ),
        ]
        answers = inquirer.prompt(questions)
        if answers['save'] == 'Yes':
            with open(force_params_path, 'w') as file:
                yaml.dump(force_params, file)   

    # selected_wrong_classification(
    #     dataset=dataset,
    #     key=key_test,
    #     spring_params=spring_params,
    #     init_range=INIT_POS_RANGE,
    #     dim=EMBEDDING_DIM,
    #     iterations=FINAL_SIM_ITERATIONS,
    #     dt=TEST_DT,
    #     damping=DAMPING,
    # )

    graph = g.to_SignedGraph(dataset, convert_to_undirected)
   
    # initialize spring state
    spring_state = sm.init_spring_state(
        rng=key_shots[shot],
        n=graph.num_nodes,
        m=graph.num_edges,
        range=INIT_POS_RANGE,
        embedding_dim=EMBEDDING_DIM
    )

    training_signs = graph.sign.copy()
    training_signs = jnp.where(graph.train_mask, training_signs, 0)
    training_graph = graph._replace(sign=training_signs)
    training_signs_one_hot = jax.nn.one_hot(training_signs + 1, 3)
    training_graph = training_graph._replace(sign_one_hot=training_signs_one_hot)

    # initialize spring state
    spring_state = sm.init_spring_state(
        rng=key_shots[shot],
        n=graph.num_nodes,
        m=graph.num_edges,
        range=INIT_POS_RANGE,
        embedding_dim=EMBEDDING_DIM
    )

    for i in range(16):
        simulation_params_test = sm.SimulationParams(
            iterations=64,
            dt=schedule_params['test_dt'],
            damping=schedule_params['test_damping'])
            
        spring_state = sm.simulate(
            simulation_params=simulation_params_test,
            spring_state=spring_state, 
            use_neural_force=use_neural_froce,
            force_params=force_params,
            graph=training_graph)

        metrics = sm.evaluate(
            spring_state,
            graph.edge_index,
            graph.sign,
            graph.train_mask,
            graph.test_mask)
        
        print(metrics)

    metrics = sm.evaluate(
        spring_state,
        graph.edge_index,
        graph.sign,
        graph.train_mask,
        graph.test_mask)
    
    print(metrics)

    # plot error per edge against avg node distance to center
    embedding_i = spring_state.position[graph.edge_index[0]]
    embedding_j = spring_state.position[graph.edge_index[1]]
    distance_i = jnp.linalg.norm(embedding_i, axis=1, keepdims=True)
    distance_j = jnp.linalg.norm(embedding_j, axis=1, keepdims=True)

    avg_distance = (distance_i + distance_j) / 2
    distance = jnp.linalg.norm(embedding_j - embedding_i, axis=1, keepdims=True)

    predicted_sign = sm.predict(
        spring_state = spring_state,
        graph = graph,
        x_0 = 0)
    print(graph.sign)
    print(predicted_sign)

    sign = jnp.where(graph.sign == 1, 1, 0)
    print(sign)
    sign_pred = jnp.where(predicted_sign > 0.01, 1, 0)
    print(sign_pred)

    true_positives = jnp.logical_and(sign == 1, sign_pred == 1)
    false_positives = jnp.logical_and(sign == 0, sign_pred == 1)
    true_negatives = jnp.logical_and(sign == 0, sign_pred == 0)
    false_negatives = jnp.logical_and(sign == 1, sign_pred == 0)

    print(f"true_positives: {jnp.sum(true_positives)}")
    print(f"false_positives: {jnp.sum(false_positives)}")
    print(f"true_negatives: {jnp.sum(true_negatives)}")
    print(f"false_negatives: {jnp.sum(false_negatives)}")

    print(f"sign: {sign}")
    print(f"sign_pred: {sign_pred}")

    error = jnp.square(sign - predicted_sign)

    plt.scatter(distance, error, c=predicted_sign, cmap='viridis', s=0.5)
    plt.colorbar()
    plt.xlabel('avg distance to center')
    plt.ylabel('error')
    plt.title('error per edge')
    plt.show()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(spring_state.position)
    embedding = pca.transform(spring_state.position)

    plt.scatter(embedding[:,0], embedding[:,1], cmap='viridis', s=0.5)
    plt.colorbar()
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA embedding')
    plt.show()

    error_per_node = jnp.zeros(graph.num_nodes)
    error_per_node = error_per_node.at[graph.edge_index[0]].add(error)
    error_per_node = jnp.expand_dims(error_per_node, axis=1)
    # normalize error by node degree, elementwise division
    error_per_node = error_per_node / graph.degree.values

    print(f"error_per_node: {error_per_node.shape}")

    plt.scatter(graph.degree.values, error_per_node, s=1.5)
    plt.xlabel('node degree')
    plt.ylabel('error')
    # plt.title(')
    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])
    