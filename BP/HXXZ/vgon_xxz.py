import os
# os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import pennylane as qml
import warnings
warnings.filterwarnings('ignore')
import os
import sys, math
import scipy.io

# Circuit ansatz
def apply_1q_gate(params, wire):
    qml.RZ(params[0], wires=wire)
    qml.RY(params[1], wires=wire)
    qml.RZ(params[2], wires=wire)

def apply_2q_gate(params, wires):
    # params of shape (5, 3)
    apply_1q_gate(params[0], wires[0])
    apply_1q_gate(params[1], wires[1])
    qml.CNOT(wires=[wires[1], wires [0]])
    qml.RZ(params[2, 0], wires=wires [0])
    qml.RY(params[2, 1],wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2, 2], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    apply_1q_gate(params[3], wires [0])
    apply_1q_gate(params[4], wires [1])

def layer_2U(input):
    idx_gate = 0
    for i in np.arange(0, nq-1):
        apply_2q_gate(input[idx_gate:idx_gate+5,:], wires=[i, (i+1)])
        idx_gate += 5

# Hamiltonian
def H_XXZ(nq,is_open):
    coeffs = []
    obs = []
    JS = [-1, -1, 1] # coupling
    for i in range(nq):
        if (is_open and i < nq-1) or (not is_open):
            obs.append(qml.PauliX(i) @ qml.PauliX((i+1)%nq))
            coeffs.append(JS[0])
            obs.append(qml.PauliY(i) @ qml.PauliY((i+1)%nq))
            coeffs.append(JS[1])
            obs.append(qml.PauliZ(i) @ qml.PauliZ((i+1)%nq))
            coeffs.append(JS[2])
    return qml.Hamiltonian(coeffs, obs)

# Pennylane, simulate quantum circuits
nq = 18
dev = qml.device("lightning.gpu", wires=nq, batch_obs=True)

@qml.qnode(dev, interface='torch', diff_method='adjoint')
def circuit(inputs): # return energiy
    qml.layer(layer_2U, nlayer, inputs)
    return qml.expval(H_XXZ(nq, False))  # False → PBC, True → OBC

@qml.qnode(dev, interface='auto')
def circuit_vec(inputs): # return final state to calculate fidelity
    qml.layer(layer_2U, nlayer, inputs)
    return qml.state()


# VGON model
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
        log_var = self.e3(h) # log of variance, easy for computing
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
        return circuit(out.transpose(0, 1).reshape(tuple(np.append(shapes, z.shape[0]))))
    
    # used to calculate fidelity
    def decoder_vec(self, z):
        out = F.relu(self.d4[0](z))
        for i in range(1, len(self.d4)):
            out = F.relu(self.d4[i](out))
        out = self.d5(out)
        return out

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var

if __name__ == '__main__':
    # # INFO
    # 18 qubits: -44.5364/18 = -2.4742

    # # PARAMETERS
    # 1. VGON
    nq = 18
    n_samples = 10 # number of fidelity
    fid_freq = 10 # frequency of computing fidelity
    epochs = 1
    batch_size = 8
    z_dim = 100
    max_iter = 1000
    kl_coeff = 0.1
    training_sample_size = 1000000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. ansatz
    nlayer = 48
    shapes = [(nlayer, 5*(nq-1), 3)]
    n_param = sum([np.prod(shape) for shape in shapes])
    list_z = np.arange(math.floor(math.log2(n_param)), math.ceil(math.log2(z_dim))-1, -1, dtype=int)
    vec_exp = np.vectorize(lambda x: 2**x)
    h_dim = np.insert(vec_exp(list_z), 0, n_param)[1:]
    learning_rate = 0.0001
    para_dim = sum([np.prod(shape) for shape in shapes])

    # # FILEs
    # 1. csv/model
    file_name = 'vgon' + '_nq' + str(nq) + '_nl' + str(nlayer) + '_' + str(h_dim) + '_lr_' + str(learning_rate) +'_latent' + str(z_dim) + '_batch' + str(batch_size) + '_KL_coeff'+ str(kl_coeff)
    model_name = os.path.join(sys.path[0], "result/model", file_name)
    csv_name = os.path.join(sys.path[0], "result/data", file_name)

    # 2. ED state
    path = os.path.join(sys.path[0], "result", "GS" + "_" + str(nq) + ".mat")
    psi = torch.tensor(scipy.io.loadmat(path)['psi_T']).to(device)
    psi = torch.reshape(psi, (-1,))

    # # INITIALIZATION
    # 1. data
    x = torch.rand(training_sample_size, para_dim) * 2 * torch.pi
    train_db = torch.utils.data.DataLoader(
        list(x), shuffle=True, batch_size=batch_size, drop_last=True
    )

    n_repeat = 10
    histories = [] # energy_min, energy_mean, kl_div
    fidelities = []

    for l in range(n_repeat):
        # 2. model
        model = Model(para_dim, z_dim, h_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # # TRAINING
        min_loss = 1e9
        min_energy = 1e9

        for epoch in range(epochs):
            for i, batch_x in enumerate(train_db):
                # 1. update parameters
                batch_x = batch_x.to(device)
                opt.zero_grad(set_to_none=True)
                out, mean, log_var = model(batch_x)
                min_energy = min(min_energy, torch.min(out))
                energy = torch.mean(out)
                kl_div = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
                kl_div = kl_div.mean()
                loss_evaluated = energy + kl_coeff * kl_div
                loss_evaluated.backward()
                opt.step()

                # 2. calculate fidelity
                if i%fid_freq == 0 or i == max_iter:
                    dist = MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim))
                    z = dist.sample([n_samples]).to(device)
                    state_param = model.decoder_vec(z)
                    params_begins_T = torch.transpose(state_param, 0, 1)
                    state_param = torch.reshape(params_begins_T, shapes[0]+tuple([n_samples]))
                    state = circuit_vec(state_param)

                    fidelity = qml.math.fidelity_statevector(state, psi,check_state=True)
                    print('mean_fid =', str(torch.mean(fidelity).item()), 'var_fid =', str(torch.var(fidelity).item()))
                    fidelities.append(fidelity.cpu().detach().numpy())

                # 3. save data/model
                histories.append([torch.min(out).cpu().detach().numpy(), energy.cpu().detach().numpy(), kl_div.cpu().detach().numpy()])
                
                print('Iteration', i, 'loss =', str(loss_evaluated.cpu().detach().numpy()), 'min energy =', str(histories[-1][1]), 'kl_div =', str(histories[-1][2]))

                if min_loss > loss_evaluated:
                    min_loss = loss_evaluated
                    torch.save(model.state_dict(), model_name + '.pth')
            
                if i >= max_iter:
                    break

        print('min_energy =', str(min_energy))

        np.savetxt(csv_name + '_history.csv', histories, delimiter=',')
        np.savetxt(csv_name + '_fidelity.csv', fidelities, delimiter=',')
        