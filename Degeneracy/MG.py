import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
import pennylane as qml
import warnings
warnings.filterwarnings('ignore')
import os
import math
import itertools


# Circuit ansatz
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

def layer_2U(input):
    idx_gate = 0
    for i in np.arange(0, nq-1):
        apply_2q_gate(input[idx_gate:idx_gate+5,:],wires=[i, (i+1)])
        idx_gate += 5

# target Hamiltonian
def H_GM(n):
    coeffs, ob = [], []
    for i in range(n-2):
        coeffs.extend([1, 1, 1])
        ob.extend([qml.PauliX(i)@qml.PauliX(i+1), qml.PauliY(i) @
                qml.PauliY(i+1), qml.PauliZ(i)@qml.PauliZ(i+1)])
        coeffs.extend([1, 1, 1])
        ob.extend([qml.PauliX(i)@qml.PauliX(i+2), qml.PauliY(i) @
                qml.PauliY(i+2), qml.PauliZ(i)@qml.PauliZ(i+2)])
        coeffs.extend([1, 1, 1])
        ob.extend([qml.PauliX(i+1)@qml.PauliX(i+2), qml.PauliY(i+1) @
                qml.PauliY(i+2), qml.PauliZ(i+1)@qml.PauliZ(i+2)])
    return qml.Hamiltonian(coeffs, ob)

# setting
nq = 10# energy = 21
dev = qml.device("default.qubit", wires=nq, shots=None)
@qml.qnode(dev, interface='torch', diff_method="best")
def circuit(inputs):
    qml.layer(layer_2U, nlayer, inputs)
    return qml.expval(H_GM(nq))  # False → PBC, True → OBC


class Model(nn.Module):
    def __init__(self, para_dim, z_dim, h_dim):
        super(Model, self).__init__()
        # encoder
        self.e1 = nn.ModuleList([nn.Linear(para_dim, h_dim[0],bias=True)])
        self.e1 += [nn.Linear(h_dim[i-1], h_dim[i],bias=True) for i in range(1, len(h_dim))]
        self.e2 = nn.Linear(h_dim[-1], z_dim,bias=True) # get mean prediction
        self.e3 = nn.Linear(h_dim[-1], z_dim,bias=True) # get mean prediction

        # decoder
        self.d4 = nn.ModuleList([nn.Linear(z_dim, h_dim[-1],bias=True)])
        self.d4 += [nn.Linear(h_dim[-i+1], h_dim[-i],bias=True) for i in range(2, len(h_dim)+1)]
        self.d5 = nn.Linear(h_dim[0], para_dim,bias=True)

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
        eps = torch.randn(log_var.shape)
        std = torch.exp(log_var).pow(0.5) # square root
        z = mean + std * eps
        return z

    def decoder_vec(self, z):
        out = F.relu(self.d4[0](z))
        for i in range(1, len(self.d4)):
            out = F.relu(self.d4[i](out))
        out = self.d5(out)
        return out

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        out = self.decoder_vec(z)
        return out, mean, log_var

if __name__ == '__main__':
    # setting
    nq = 10# energy = 21
    lim = -(nq-2)*3 + 0.1

    # param
    batch_size = 50
    z_dim = 50
    kl_coeff = 1
    e_coeff = 1

    nlayer = 4
    shapes = [(nlayer, 5*(nq-1), 3)]
    n_param = sum([np.prod(shape) for shape in shapes])

    # training settings
    num_iter = 1500
    training_sample_size = 160000
    cos_sim_setting = [20, 20, 20, 10, 10, 10, 5, 5, 5, 2, 2, 2]
    list_z = np.arange(math.floor(math.log2(2*n_param)),math.ceil(math.log2(z_dim))-1, -1, dtype=int)
    vec_exp = np.vectorize(lambda x: 2**x)
    h_dim = vec_exp(list_z)
    learning_rate = 0.0014

    data_dist = dists.Uniform(0, 1)    #uniform
    x = data_dist.sample([training_sample_size, n_param])
    train_db = torch.utils.data.DataLoader(
        list(x), shuffle=True, batch_size=batch_size, drop_last=True
    )

    # model
    model = Model(n_param, z_dim, h_dim)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)


    loss_training = []
    cos_sim_training = []


    for i, batch_x in enumerate(train_db):
        # # training phase
        model.train()
        batch_x = batch_x
        opt.zero_grad(set_to_none=True)
        out, mean, log_var = model(batch_x)

        # loss function
        # 1. energy
        out_energy = circuit(out.transpose(0,1).reshape(tuple(np.append(shapes,batch_size))))
        
        # 2. kl
        kl_div = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
        kl_div = kl_div.mean()

        
        #  cos-similarity among a batch 
        cos_sims = torch.empty((0))
        for idx in itertools.combinations(range(batch_size), 2):
            cos_sim = torch.cosine_similarity(out[idx[0], :], out[idx[1], :], dim=0)
            cos_sims = torch.cat((cos_sims, cos_sim.unsqueeze(0)), dim=0)
        avg_cos_sims = cos_sims.mean()

        coeffs_list = cos_sim_setting + [1] * 20
        cos_sim_coeff = coeffs_list[int(i//100)]

        loss_evaluated = e_coeff*torch.mean(out_energy) + kl_coeff * kl_div + cos_sim_coeff*avg_cos_sims
        loss_evaluated.backward()
        opt.step()

        if torch.mean(out_energy).item() <= lim or i >= num_iter:
            torch.save(model.state_dict(), 'degeneracy_model' + '.pth')


        
 