# plot loss over time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('output/neural_training_results_20.csv')
print(data)
epochs = np.arange(0, len(data.loss))
plt.plot(epochs, data.loss)
plt.title('Loss')
plt.show()

# get set1 color map
#colors = plt.cm.get_cmap('Set1', 10)
colors = ['#d73027', '#4575b4', '#91bfdb', '#984ea3', '#ff7f00']
# Three subplots with shared x axis
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
# plot metrics over time
ax1.plot(epochs, data.auc, color=colors[0])
ax1.plot(epochs, data.f1_macro, color=colors[1])
# shared x axis
# set legend position top right

ax1.legend(['AUC', 'F1 macro'], loc='lower right')

ax2.plot(epochs, data.loss, color=colors[0])
ax2.legend(['loss'], loc='upper right')

ax3.plot(epochs, data.dt, color=colors[0])
ax3.plot(epochs, data.damping, color=colors[1])

ax3.legend(['dt', 'damping'], loc='upper right')

ax4.plot(epochs, data.iterations, color=colors[0])
ax4.legend(['iterations'], loc='lower right')
ax4.set_xlabel('epochs')

plt.show()