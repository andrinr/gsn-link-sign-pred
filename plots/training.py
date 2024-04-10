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
colors = ['#EB0065', '#EAD100', '#00C5EB', '#984ea3', '#ff7f00']
# Three subplots with shared x axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# plot metrics over time
ax1.plot(epochs, data.auc, color=colors[0])
ax1.plot(epochs, data.f1_macro, color=colors[2])
ax1.legend(['AUC', 'F1 macro'], loc='lower left')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')

ax2.plot(epochs, data.dt, color=colors[0])
ax2.plot(epochs, data.damping, color=colors[2])
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Value')

ax2.legend(['dt', 'damping'], loc='lower left')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Value')

ax3.plot(epochs, data.iterations, color=colors[0])
ax3.legend(['iterations'], loc='lower left')
ax3.set_xlabel('Epochs')

plt.show()