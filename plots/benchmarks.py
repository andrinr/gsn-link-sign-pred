# load all files from benchmarks folder

import os
import pandas as pd

# get all files in the directory
files = os.listdir('plots/data/benchmarks')

# load all csv files and append to pandas dataframe
dfs = []
for file in files:
    df = pd.read_csv(f'plots/data/benchmarks/{file}')
    # remove first column
    df = df.drop(df.columns[0], axis=1)
    dfs.append(df)

# concatenate all dataframes
data = pd.concat(dfs)

value_columns = data.columns[0:6]
std_columns = data.columns[6:12]
description_columns = data.columns[12:15]

# multiply values and stds by 100
data[value_columns] = data[value_columns] * 100
data[std_columns] = data[std_columns] * 100

# set to two decimal places
data[std_columns] = data[std_columns].round(2)
data[value_columns] = data[value_columns].round(2)

# write to csv
data.to_csv('plots/data/benchmarks.csv', index=False)


