# plot loss over time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee'])

# load data
data = pd.read_csv(
    'plots/data/neural_forward_euler_results_20.csv', 
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
print(data)

fig, ax = plt.subplots()


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(data.iterations, data.auc, label='auc')
ax1.plot(data.iterations, data.f1_binary, label='f1_binary')
ax1.plot(data.iterations, data.f1_micro, label='f1_micro')
ax1.plot(data.iterations, data.f1_macro, label='f1_macro')

# put legend outside of the plot    
ax1.legend(loc='lower right', ncol=2)

ax2.plot(data.iterations, data.mean_velocity, label='mean velocity')

ax2.set_xlabel('Iterations')
ax2.legend()


plt.savefig('plots/forward.png')