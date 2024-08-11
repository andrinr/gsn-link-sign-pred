import inquirer
import getopt
from data import Slashdot, BitcoinO, BitcoinA, WikiRFA, Epinions, Tribes
from torch_geometric.data import Data 
import torch_geometric.transforms as T
    
def get_dataset(data_path : str, dataset_name : str) -> Data:

    pre_transform = T.Compose([])
    match dataset_name:
        case "BitcoinOTC":
            dataset = BitcoinO(root=data_path, pre_transform=pre_transform)
        case "BitcoinAlpha":
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

    transform = T.ToUndirected(reduce="min")
    data = transform(data)

    return data