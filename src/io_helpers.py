import inquirer
import getopt
from data import Slashdot, BitcoinO, BitcoinA, WikiRFA, Epinions, Tribes
from torch_geometric.data import Data 
import torch_geometric.transforms as T
    
def get_dataset(data_path : str, argv : list) -> Data:
    dataset_names = ['Tribes', 'Bitcoin_Alpha', 'BitcoinOTC', 'WikiRFA', 'Slashdot', 'Epinions']
    questions = [
        inquirer.List('dataset',
            message="Choose a dataset",
            choices=dataset_names,
        ),
    ]
    answers = inquirer.prompt(questions)
    dataset_name = answers['dataset']

    opts,batch_index = getopt.getopt(argv,"s:h:d:i:p:o",
        ["embedding_size=","time_step=", "damping=", "iterations="])
    for opt, arg in opts:
        if opt == '-s':
            EMBEDDING_DIM = int(arg)
        elif opt == '-h':
            DT = int(arg)
        elif opt == '-d':
            DAMPING = int(arg)
        elif opt == '-i':
            FINAL_SIM_ITERATIONS = int(arg)

    pre_transform = T.Compose([])
    match dataset_name:
        case "BitcoinOTC":
            dataset = BitcoinO(root=data_path, pre_transform=pre_transform)
        case "Bitcoin_Alpha":
            dataset = BitcoinA(root=data_path, pre_transform=pre_transform)
        case "WikiRFA":
            dataset = WikiRFA(root=data_path, pre_transform=pre_transform)
        case "Slashdot":
            dataset = Slashdot(root=data_path, pre_transform=pre_transform)
        case "Epinions":
            dataset = Epinions(root=data_path, pre_transform=pre_transform)
        case "Tribes":
            dataset = Tribes(root=data_path, pre_transform=pre_transform)

    data = dataset[0]

    return data