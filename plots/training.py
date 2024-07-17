# plot loss over time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee'])

# set font style to roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


# load data
data = pd.read_csv('plots/data/neural_training_results_20.csv')
print(data)
epochs = np.arange(0, len(data.loss))
epochs = epochs[0:200]

# get set1 color map
#colors = plt.cm.get_cmap('Set1', 10)
colors = ['#d73027', '#4575b4', '#91bfdb', '#984ea3', '#ff7f00']
# Three subplots with shared x axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# plot metrics over time
ax1.plot(epochs, data.auc[0:200])#, color=colors[0])
ax1.plot(epochs, data.f1_macro[0:200])#, color=colors[1])
# shared x axis
# set legend position top right

ax1.legend(['AUC', 'F1 macro'], loc='lower right')

ax2.plot(epochs, data.loss[0:200])#, color=colors[0])
ax2.legend(['loss'], loc='upper right')
# set y log
# plt.yscale('log')
# set x label
plt.xlabel('Epochs')

# store plots
plt.savefig('plots/training.png')
