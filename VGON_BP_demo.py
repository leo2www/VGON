import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
import pennylane as qml
from itertools import product
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
import os
import sys, math
import scipy.io as sio
import datetime
import logging
import collections
import copy

# # INFO
# 14 qubits: -57.2478/14 = -4.0891; m2: -61.7993/14 = -4.4142

# # PARAMETERS
epochs = 1000
max_iter=100
batch_size = 4
nq = 6
ansatz_index = '2U' #chose ansatz(random or 2U)
mpd_layers= 4
ham = "HXX"
JS = [1,1] # coupling
z_dim = 8
kl_coeff = 0.1
vec_exp=np.vectorize(lambda x: 2**x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dev = qml.device("lightning.gpu", wires=nq, batch_obs=True)
dev = qml.device('default.qubit', wires=nq)
time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# cuda_id = torch.cuda.current_device()

# # CHOOSE ANSATZ
# 1. ansatz - 2U
def apply_1q_gate(params, wire):
    qml.RZ(params[0], wires=wire)
    qml.RY(params[1], wires=wire)
    qml.RZ(params[2], wires=wire)

def apply_2q_gate(params, wires):
    # params of shape (5, 3)
    apply_1q_gate(params[0, :], wires[0])
    apply_1q_gate(params[1, :], wires[1])
    qml.CNOT(wires=[wires[1], wires [0]])
    qml.RZ(params[2, 0], wires=wires [0])
    qml.RY(params[2, 1],wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2, 2], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    apply_1q_gate(params[3, :], wires [0])
    apply_1q_gate(params[4, :], wires [1])

def layer_2U(input,rot_idx):
    idx_gate = 0
    for i in np.arange(0, nq-1):
        apply_2q_gate(input[idx_gate:idx_gate+5,:],wires=[i, (i+1)])
        idx_gate += 5

# 2. ansatz - 2U
# np.random.seed(923)
rot_gate = [qml.RX, qml.RY, qml.RZ]
def layer_random(inputs,rot_idx):
    for i in range(nq):
        rot_gate[rot_idx[i]](inputs[i], wires=i)
    if nq > 1:
        for i in range(nq):
            qml.CZ(wires=[i, (i+1)%nq])

# 3. ansatz - selection
if ansatz_index == 'random':
    #ansatz parameters
    nlayer = 300
    shapes = [(nlayer, nq)]
    # h_dim = [sum([np.prod(shape) for shape in shapes]), 2048, 1024, 512, 256, 128, 64]#  h_dim 1
    h_dim = [sum([np.prod(shape) for shape in shapes]), 2048, 1024, 512, 256, 128, 64][1:]#  h_dim 1
    layer_1 = layer_random
    learning_rate = 0.0001
    rot_idx = np.random.choice([0, 1, 2], (nlayer, nq), replace=True)
elif ansatz_index == '2U':
    #ansatz parameters
    nlayer = mpd_layers
    shapes = [(nlayer, 5*(nq-1), 3)]
    n_param = sum([np.prod(shape) for shape in shapes])
    # list_z = np.arange(math.ceil(math.log2(n_param)),math.ceil(math.log2(z_dim))-1, -1, dtype=int)
    list_z = np.arange(math.floor(math.log2(n_param)),math.ceil(math.log2(z_dim))-1, -1, dtype=int)
    #h_dim = np.insert(vec_exp(list_z),0,n_param)
    h_dim = vec_exp(list_z)
    layer_1 = layer_2U
    learning_rate=0.0005
    rot_idx = np.random.choice([0, 1, 2], (nlayer, nq), replace=True)

para_dim = sum([np.prod(shape) for shape in shapes])

# 4. ansatz - pennylane circuit
def H_XX(nq,is_open):
    coeffs = []
    obs = []
    JS = [1,1] # coupling
    for i in range(nq):
        if (is_open and i < nq-1) or (not is_open):
            obs.append(qml.PauliX(i) @ qml.PauliX((i+1)%nq))
            coeffs.append(JS[0])
            obs.append(qml.PauliY(i) @ qml.PauliY((i+1)%nq))
            coeffs.append(JS[1])
    return qml.Hamiltonian(coeffs, obs)

@qml.qnode(dev, interface='torch', diff_method='adjoint')
def circuit(inputs):
    inputs = torch.reshape(inputs, shapes[0])
    qml.layer(layer_1, nlayer, inputs, rot_idx)
    return qml.expval(H_XX(nq, False))  # False → PBC, True → OBC

@qml.qnode(dev, interface='auto')
def circuit_vec(inputs):
    qml.layer(layer_1, nlayer, inputs, rot_idx)
    return qml.state()

# # VAE MODEL
class Model(nn.Module):
    def __init__(self, para_dim, z_dim, h_dim):
        super(Model, self).__init__()
        # encoder
        self.e1 = nn.ModuleList([nn.Linear(para_dim, h_dim[0])])
        self.e1 += [nn.Linear(h_dim[i-1], h_dim[i]) for i in range(1, len(h_dim))]
        self.e2 = nn.Linear(h_dim[-1], z_dim) # get mean prediction
        self.e3 = nn.Linear(h_dim[-1], z_dim) # get mean prediction

        # batch normalization
        self.bn = torch.nn.BatchNorm1d(z_dim)

        # decoder
        self.d4 = nn.ModuleList([nn.Linear(z_dim, h_dim[-1])])
        self.d4 += [nn.Linear(h_dim[-i+1], h_dim[-i]) for i in range(2, len(h_dim)+1)]
        self.d5 = nn.Linear(h_dim[0], para_dim)

    def encoder(self, x):
        h = F.relu(self.e1[0](x))
        for i in range(1, len(self.e1)):
            h = F.relu(self.e1[i](h))
        # get_mean
        mean = self.e2(h)
        # get_variance
        log_var = self.e3(h)    # log of variance, easy for computing'result/model/vae_' + str(nq) + '_' + str(nlayer) + '.pth'
        return mean, log_var

    def reparameterize(self, mean, log_var):
        eps = torch.randn(log_var.shape).to(device)
        std = torch.exp(log_var).pow(0.5) # square root
        z = mean + std * eps
        return z

    def decoder(self, z):
        out = F.relu(self.d4[0](z))
        for i in range(1, len(self.d4)):
            out = F.relu(self.d4[i](out))
        out = self.d5(out)

        _out = []
        for i in range(out.shape[0]):
            _out.append(circuit(out[i]))
        out = torch.vstack(_out)
        return out

    # def decoder(self, z):
    #     out = F.elu(self.d4[0](z))
    #     for i in range(1, len(self.d4)):
    #         out = F.elu(self.d4[i](out))
    #     out = self.d5(out)
    #     return circuit(out.transpose(0,1).reshape(tuple(np.append(shapes,batch_size))))
   
    
    def decoder_vec(self, z):
        out = F.relu(self.d4[0](z))
        for i in range(1, len(self.d4)):
            out = F.relu(self.d4[i](out))
        out = self.d5(out)
        return out

    def forward(self, x):
        mean, log_var = self.encoder(x)
        # trick: reparameterization trick, sampling
        # mean = self.bn(mean)
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var

# # FILEs
# 1. log
log_name = 'vae' + ham + '_nq' + str(nq) + '_nl' + str(nlayer) + '_' + str(ansatz_index) + '_' + str(h_dim) + '_lr_' + str(learning_rate) + '_latent' + str(z_dim) + '_batch' + str(batch_size) + 'KL_coeff' + str(kl_coeff)+'_E0E1.log'
path_log = os.path.join(sys.path[0], 'logs', log_name)
logging.basicConfig(format = '%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',encoding='utf-8', level=logging.INFO, handlers=[logging.FileHandler(path_log), logging.StreamHandler(sys.stdout)])
logging.info(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
logging.info(f"CUDA version: {torch.version.cuda}")
# logging.info(f"CUDA device: {torch.cuda.get_device_name(cuda_id)}")
logging.info('%s', circuit.device)
logging.info('%s %d', 'Number of qubits: ', nq)
logging.info('%s %s', 'Ansatz: ', ansatz_index)
logging.info('%s %s', 'Number of layers and qubits: ', str(shapes))
logging.info('%s %d', 'Total number of parameters: ', para_dim)
logging.info('%s %s', 'Netwoork layer: ', str(h_dim))
logging.info('%s %s', 'Learning rate: ', str(learning_rate))
logging.info('%s %s', 'KL coeff:', str(kl_coeff))
logging.info('%s %d', 'latent dim: ', z_dim)
logging.info('%s %d', 'batch size: ', batch_size)

# 2. csv/model
file_name = 'vae' + ham + '_nq' + str(nq) + '_nl' + str(nlayer) + '_' + str(ansatz_index) + '_' + str(h_dim) + '_lr_' + str(learning_rate) +'_latent' + str(z_dim) + '_batch' + str(batch_size) + '_KL_coeff'+ str(kl_coeff)
model_name = os.path.join(sys.path[0], "result/model", file_name )
csv_name = os.path.join(sys.path[0], "result", file_name )
logging.info('%s %s', 'File Name: ', str(file_name))


# # INITIALIZATION
# 1. data
data_dist=dists.VonMises(0,torch.pi)
x=data_dist.sample([batch_size*30, para_dim])
train_db = torch.utils.data.DataLoader(
    list(x), shuffle=True, batch_size=batch_size, drop_last=True
)
# 2. model
model = Model(para_dim, z_dim, h_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # TRAINING
min_loss = 1e9
min_energy = 1e9
vae_history = []


for epoch in range(epochs):
    for i, batch_x in enumerate(train_db):
        # 1. update parameters
        batch_x = batch_x.to(device)
        opt.zero_grad(set_to_none=True)
        out, mean, log_var = model(batch_x)
        # out /= nq
        min_energy = min(min_energy, torch.min(out))
        energy = torch.mean(out)
        kl_div = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
        kl_div = kl_div.mean()
        loss_evaluated = energy + kl_coeff * kl_div
        loss_evaluated.backward()
        opt.step()


        # 2. save data/model
        vae_history.append([torch.min(out).cpu().detach().numpy(), energy.cpu().detach().numpy(), kl_div.cpu().detach().numpy()])
        
        logging.info('%s %d %s %d %s %s %s %s %s %s', 'Epoch',epoch,'Iteration', i, 'loss =', str(loss_evaluated.cpu().detach().numpy()), 'min energy =', str(vae_history[-1][1]), 'kl_div =', str(vae_history[-1][2]))
       
    
        if i >= max_iter:
            break

logging.info('%s %s', 'min_energy = ', str(min_energy))
