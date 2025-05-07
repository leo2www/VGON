import numpy as np
import os, sys, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import matplotlib.colors as mcolors


# PARAS
nq = 18
nlayer = 48
lr = 0.01
ED_v = -1.7828
fid_freq = 10 # frequency of computing fidelity
path = sys.path[0]

# VQE - DATA
file_name = 'vqe' + '_nq' + str(nq) + '_nl' + str(nlayer) + '_lr' + str(lr) + '_energy.csv'
file_name = os.path.join(path, "result/data", file_name)
energies = np.genfromtxt(file_name, delimiter=',')
energies = energies[:, [i*fid_freq for i in range(int(energies.shape[1]/fid_freq))]]

print('VQE, min energy', energies.min())
print('VQE, mean energy density', energies[:, -1].mean()/nq)

n_loop, n_iteration = energies.shape
data_VQE = []
for loop in range(n_loop):
    for iteration in range(n_iteration):
        data_VQE.append({'Loop': loop, 'Iteration': iteration*fid_freq+1, 'Energy Density': energies[loop, iteration]/nq})
data_VQE = pd.DataFrame(data_VQE)

#  VQE-SA - DATA
file_name = 'vqe' + '_nq' + str(nq) + '_nl' + str(nlayer) + '_lr' + str(lr) + '_init1e-2'
file_name = os.path.join(path, "result/data", file_name)

energies = np.genfromtxt(file_name + '_energy.csv', delimiter=',')
energies = energies[:, [i*fid_freq for i in range(int(energies.shape[1]/fid_freq))]]
fidelities = np.genfromtxt(file_name + '_fidelity.csv', delimiter=',')
fidelities = fidelities[:, [i*fid_freq for i in range(int(fidelities.shape[1]/fid_freq))]]

print('SA, min energy', energies.min())
print('SA, mean energy density', energies[:, -1].mean()/nq)

n_loop, n_iteration = energies.shape
data_SA = []
for loop in range(n_loop):
    for iteration in range(n_iteration):
        data_SA.append({'Loop': loop, 'Iteration': iteration*10+1, 'Energy Density': energies[loop, iteration]/nq, 'Fidelity': fidelities[loop, iteration]})
data_SA = pd.DataFrame(data_SA)

# VGON - DATA
# # PARAMETERS
batch_size = 8
z_dim = 100
kl_coeff = 0.1
nlayer = 48
shapes = [(nlayer, 5*(nq-1), 3)]
n_param = sum([np.prod(shape) for shape in shapes])
list_z = np.arange(math.floor(math.log2(n_param)), math.ceil(math.log2(z_dim))-1, -1, dtype=int)
vec_exp = np.vectorize(lambda x: 2**x)
h_dim = np.insert(vec_exp(list_z), 0, n_param)[1:]
learning_rate = 0.0001

file_name = 'vgon' + '_nq' + str(nq) + '_nl' + str(nlayer) + '_' + str(h_dim) + '_lr_' + str(learning_rate) +'_latent' + str(z_dim) + '_batch' + str(batch_size) + '_KL_coeff'+ str(kl_coeff)
file_name = os.path.join(sys.path[0], "result/data", file_name)
_energies = np.genfromtxt(file_name + '_history.csv', delimiter=',')[:, 0]
n_loop = 1001
energies = _energies[:n_loop].reshape(-1, 1)
for i in range(1, int(_energies.shape[0] // n_loop)):
    energies = np.concatenate((energies, _energies[i*n_loop:(i+1)*n_loop].reshape(-1, 1)), axis=1)

_fidelities = np.genfromtxt(file_name + '_fidelity.csv', delimiter=',')
cnt = 101
fidelities = _fidelities[:cnt, :]
for i in range(1, int(_fidelities.shape[0] // cnt)):
    fidelities = np.concatenate((fidelities, _fidelities[i*cnt:(i+1)*cnt, :]), axis=1)
fidelities = np.transpose(fidelities)

print('VGON, mean energy density', energies[-1, :].mean()/nq)

n_loop, n_iteration = (fidelities).shape
data_VGON = []
for loop in range(n_loop):
    for iteration in range(n_iteration):
        data_VGON.append({'Loop': loop, 'Iteration': iteration*fid_freq+1, 'Energy Density': energies[iteration*fid_freq, int(loop//fid_freq)]/nq, 'Fidelity': fidelities[loop, iteration]})
data_VGON = pd.DataFrame(data_VGON)

# PLOT

f, ax1 = plt.subplots()
sns.lineplot(x='Iteration', y='Energy Density', data=data_VQE, ax=ax1, color='#38E3FF')
sns.lineplot(x='Iteration', y='Energy Density', data=data_SA, ax=ax1, color='#324675')
sns.lineplot(x='Iteration', y='Energy Density', data=data_VGON, ax=ax1, color='green')
hline = ax1.hlines(y = ED_v, xmin = -1, xmax = 1001, ls = ':', colors = 'black')
ax1.text(800, ED_v+0.05, 'ED = '+ str(ED_v), ha='center', va='bottom')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Average Energy')

ax2 = plt.axes([0.5, 0.3, .3, .2])
sns.lineplot(x='Iteration', y='Energy Density', data=data_SA, ax=ax2, color='#324675')
sns.lineplot(x='Iteration', y='Energy Density', data=data_VGON, ax=ax2, color='green')
ax2.hlines(y = ED_v, xmin = -1, xmax = 1001,ls = ':', colors = 'black')
ax2.set_ylim([-1.79, -1.73])
ax2.set_xlim([500, 1000])
ax2.set_ylabel('Average Energy')

ax3 = ax1.twinx()
sns.lineplot(x='Iteration', y='Fidelity', data=data_SA, ax=ax3, color='#AB2A3C')
sns.lineplot(x='Iteration', y='Fidelity', data=data_VGON, ax=ax3, color='#660874')
ax3.hlines(y = 0.99, xmin = -1, xmax = n_loop, ls = '--', colors = 'black', label='Fidelity=0.99') 
ax3.text(120, 0.97, 'Fidelity = 0.99', ha='center', va='top')
ax3.set_ylabel('Fidelity')
plt.grid()
legend_energy = ax2.legend([ax1.lines[0], ax1.lines[1], ax1.lines[2], hline, ax3.lines[0], ax3.lines[1]], 
                           ['Average Energy, VQE', 'Average Energy, VQE-SA', 'Average Energy, VGON', 'Average Energy, Exact',
                            'Fidelity, VQE-SA', 'Fidelity, VGON'], loc='upper center', bbox_to_anchor=(0, 3.5), ncol=3, frameon=False)
ax2.add_artist(legend_energy)


name_fig = os.path.join(sys.path[0], "result/figure", 'bp.pdf')
f.savefig(name_fig)
