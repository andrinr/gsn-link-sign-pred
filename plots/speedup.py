# plot loss over time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee'])

# load data
data = pd.read_csv(
    'plots/data/speedup.csv', 
    header=0, delimiter=',', skipinitialspace=True,
    index_col=0)
# DATAFRAME
#                 BitcoinAlpha,   BitcoinOTC,     WikirRFA,   Slashdot,   Epinions
# SpringE (JIT),  350,            354,            360,        397,        506
# SE-NN (JIT),    700,            845,            745,        1300,       1100
# SpringE,        8740,           10560,          9448,       10628,      14211
# SE-NN,          20700,          10560,          23700,      29981,      33370
# SGCN,           8089,           9791,           24355,      6000000,    21000000
# SDGNN,          388000,        818250,          2618500,

# transpose data
data = data.T

fig, ax = plt.subplots()

# set font style to roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# plasma colormap
colors = plt.cm.get_cmap('plasma', len(data.columns))
colors = list(colors.colors)
# add black stroke to colors
barplot = data.plot(kind='bar', ax=ax, logy=True, width=0.9, color=colors, edgecolor='black', linewidth=0.5)


# set grid to background
ax.grid(axis='y', linestyle='-', alpha=1.0, linewidth=0.5, color='black')

# set labels
ax.set_ylabel('Runtime (ms)')

# set legend outside of plot
# set text size to 10point and font to roman
ax.legend(title='Method', loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title_fontsize=10, facecolor='white', framealpha=1)

# tilt x labels
plt.xticks(rotation=45)

plt.savefig('plots/performance.png')