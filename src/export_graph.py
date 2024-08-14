from io_helpers import get_dataset
import pandas as pd
import torch_geometric


epinions = get_dataset('src/data/', 'Slashdot')

transforms = torch_geometric.transforms.Compose([torch_geometric.transforms.ToUndirected(reduce='min')])
epinions = transforms(epinions)

signs = epinions.edge_attr.numpy()
# create df with signs and edge indices

# df = pd.DataFrame()

# # add edge index
# df['id1'] = epinions.edge_index[0].numpy()
# df['id2'] = epinions.edge_index[1].numpy()
# df['sign'] = signs

# # export 
# df.to_csv('other_methods/SGCN/input/slashdot.csv', index=False)
# print('done')

num_edges = epinions.edge_index.shape[1]
edge_index_train = epinions.edge_index[:, 0:int(0.8*num_edges)]
edge_index_test = epinions.edge_index[:, int(0.8*num_edges):]
signs_train = signs[0:int(0.8*num_edges)]
signs_test = signs[int(0.8*num_edges):]

df_train = pd.DataFrame()

# add edge index
df_train['id1'] = edge_index_train[0].numpy()
df_train['id2'] = edge_index_train[1].numpy()
df_train['sign'] = signs_train

# df = pd.DataFrame()

# # add edge index
# df['id1'] = edge_index_test[0].numpy()  
# df['id2'] = edge_index_test[1].numpy()
# df['sign'] = signs_test

# remove column headers from df

# export 
# df_train.to_csv('other_methods/SiGAT/experiment-data/slashdot-test-1.edgelist', index=False, header=False, sep=' ')
# df.to_csv('other_methods/SiGAT/experiment-data/slashdot-train-1.edgelist', index=False, header=False, sep=' ')

df_train.to_csv('other_methods/snea/slashdot.csv', index=False, header=False, sep=' ')
print('done')
