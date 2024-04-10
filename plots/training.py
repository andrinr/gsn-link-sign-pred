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
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# plot metrics over time
ax1.plot(epochs, data.auc, color=colors[0])
ax1.plot(epochs, data.f1_macro, color=colors[1])
# set legend position top right

ax1.legend(['AUC', 'F1 macro'], loc='lower right')
ax1.set_xlabel('epochs')
ax1.set_ylabel('accuracy')

ax2.plot(epochs, data.dt, color=colors[0])
ax2.plot(epochs, data.damping, color=colors[1])
ax2.set_xlabel('epochs')
ax2.set_ylabel('values')

ax2.legend(['dt', 'damping'], loc='upper right')
ax2.set_xlabel('epochs')
ax2.set_ylabel('values')

ax3.plot(epochs, data.iterations, color=colors[0])
ax3.legend(['iterations'], loc='lower right')
ax3.set_xlabel('epochs')
ax3.set_ylabel('values')

plt.show()