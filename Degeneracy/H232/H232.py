
import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
import pennylane as qml
from itertools import product
import warnings
warnings.filterwarnings('ignore')
import os
import itertools
import scipy.io as sio

# -------------------------
# Quantum Circuit Components
# -------------------------
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
    for i in np.arange(0,nq-1):
        apply_2q_gate(input[idx_gate:idx_gate+5,:],wires=[i, (i+1)])
        idx_gate += 5


# target Hamiltonian
def H_232(nq, is_open):
    Js = coupling
    
    Paulis = [qml.PauliX, qml.PauliY, qml.PauliZ]
    Paulis = [[qml.PauliX], [qml.PauliY], [qml.PauliZ]] + list(product(Paulis, repeat=2))
    coeffs = []
    obs = []
    for i in range(nq):
        for idx_Pauli in range(len(Paulis)):
            if Js[idx_Pauli] == 0:
                continue
            if len(Paulis[idx_Pauli]) == 1:
                obs.append(Paulis[idx_Pauli][0](i))
                coeffs.append(Js[idx_Pauli])
            else:
                if (is_open and i < nq-1) or (not is_open):
                    obs.append(Paulis[idx_Pauli][0](i) @ Paulis[idx_Pauli][1]((i+1)%nq))
                    coeffs.append(Js[idx_Pauli])
        
    return qml.Hamiltonian(coeffs, obs)



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
        log_var = self.e3(h)  
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
    

# -------------------------
# Quantum Device & Circuit
# -------------------------
nq = 11
nlayer = 6
dev = qml.device("default.qubit", wires=nq)

@qml.qnode(dev, interface='torch')
def circuit(inputs):
    qml.layer(layer_2U, nlayer, inputs)
    return qml.expval(H_232(nq, is_open=True))  # Open boundary condition



# -------------------------
# Main Training Routine
# -------------------------

if __name__ == '__main__':
    shapes = [(nlayer, 5 * (nq - 1), 3)]
    n_param = sum(np.prod(s) for s in shapes)

    # Load ED data
    ED = sio.loadmat(os.path.join(sys.path[0], 'results',f'Site_{nq}_degenerate_232.mat'))
    basis = torch.tensor(ED['psi'])
    e_ground = ED['Energy_density'][0, 0]
    coupling = ED['J'][0]

    # Training settings
    training_sample_size = 160000
    batch_size = 50
    z_dim = 50
    learning_rate = 0.0015
    kl_coeff = 1
    e_coeff = 1
    num_iter = 1500
    lim = e_ground + 0.1
    cos_sim_schedule = [40, 40, 20, 10, 5, 2]

    h_dim = [512, 256, 128, 64]

    # Data loader
    data_dist = dists.Uniform(0, 1)
    data = data_dist.sample([training_sample_size, n_param])
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize model
    model = Model(n_param, z_dim, h_dim)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for step, batch in enumerate(train_loader):
        if step > num_iter:
                break

        model.train()
        opt.zero_grad()

        out, mean, log_var = model(batch)
        out_energy = circuit(out.transpose(0,1).reshape(tuple(np.append(shapes,batch_size))))

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / batch_size

        # Cosine similarity
        cos_sim = torch.stack([
            torch.cosine_similarity(out[i], out[j], dim=0)
            for i, j in itertools.combinations(range(batch_size), 2)
        ]).mean()

        cos_coeff = cos_sim_schedule[min(step // 100, len(cos_sim_schedule) - 1)]
        total_loss = e_coeff * out_energy.mean() + kl_coeff * kl_loss + cos_coeff * cos_sim

        total_loss.backward()
        opt.step()

        if out_energy.mean().item() <= lim:
            os.makedirs('results', exist_ok=True)
            torch.save(model.state_dict(),  os.path.join('results', 'Degeneracy232_model.pth'))
            break
    
        print(f"Step: {step}, Energy: {out_energy.mean().item():.4f}, Cos Similarity: {cos_sim.item():.4f}, KL: {kl_loss.item():.4f}")



        
 











