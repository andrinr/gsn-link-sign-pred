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

# transpose data
# remove WikiRFA column
print(data)
data = data.drop(columns=['WikiRFA'])
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