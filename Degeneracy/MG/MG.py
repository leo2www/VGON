import os
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
import pennylane as qml

# ------------------------
# Quantum Circuit Definition
# ------------------------

def apply_1q_gate(params, wire):
    qml.RZ(params[0], wires=wire)
    qml.RY(params[1], wires=wire)
    qml.RZ(params[2], wires=wire)

def apply_2q_gate(params, wires):
    apply_1q_gate(params[0], wires[0])
    apply_1q_gate(params[1], wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[2, 0], wires=wires[0])
    qml.RY(params[2, 1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2, 2], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    apply_1q_gate(params[3], wires[0])
    apply_1q_gate(params[4], wires[1])

def layer_2U(inputs):
    for i in range(nq - 1):
        idx = i * 5
        apply_2q_gate(inputs[idx:idx+5], wires=[i, i + 1])


def H_GM(n):
    coeffs, obs = [], []
    for i in range(n - 2):
        for offset in [(0, 1), (0, 2), (1, 2)]:
            for pauli in [qml.PauliX, qml.PauliY, qml.PauliZ]:
                obs.append(pauli(i + offset[0]) @ pauli(i + offset[1]))
                coeffs.append(1)
    return qml.Hamiltonian(coeffs, obs)

# ------------------------
# Quantum Device and QNode
# ------------------------

nq = 10
nlayer = 4
dev = qml.device("default.qubit", wires=nq, shots=None)

@qml.qnode(dev, interface='torch', diff_method="best")
def circuit(inputs):
    qml.layer(layer_2U, nlayer, inputs)
    return qml.expval(H_GM(nq))

# ------------------------
# VGON Model Definition
# ------------------------
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
    
# ------------------------
# Training Loop
# ------------------------

if __name__ == '__main__':
    # Parameters
    batch_size = 50
    z_dim = 50
    kl_coeff = 1
    e_coeff = 1
    lim = -(nq - 2) * 3 + 0.1
    shapes = [(nlayer, 5 * (nq - 1), 3)]
    n_param = sum(np.prod(shape) for shape in shapes)
    num_iter = 1500
    training_sample_size = 160000
    cos_sim_schedule = [40] * 1 + [20] * 3 + [5] * 3 + [2] * 3 + [1] * 20

    list_z = np.arange(math.floor(math.log2(2 * n_param)), math.ceil(math.log2(z_dim)) - 1, -1)
    h_dim = 2 ** list_z
    learning_rate = 0.0014

 # Data
    data = dists.Uniform(0, 1).sample([training_sample_size, n_param])
    train_loader = torch.utils.data.DataLoader(list(data), batch_size=batch_size, shuffle=True, drop_last=True)

    # Model & Optimizer
    model = Model(n_param, z_dim, h_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for step, batch_x in enumerate(train_loader):
        if step > num_iter:
            break

        model.train()
        optimizer.zero_grad()

        out, mean, log_var = model(batch_x)

        # Quantum energy
        out_energy = circuit(out.transpose(0,1).reshape(tuple(np.append(shapes,batch_size))))

        # KL divergence
        kl_div = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).mean()

        # Cosine similarity
        cos_sims = torch.stack([
            torch.cosine_similarity(out[i], out[j], dim=0)
            for i, j in itertools.combinations(range(batch_size), 2)
        ])
        avg_cos_sim = cos_sims.mean()

        cos_coeff = cos_sim_schedule[min(step // 100, len(cos_sim_schedule) - 1)]
        loss = e_coeff * out_energy.mean() + kl_coeff * kl_div + cos_coeff * avg_cos_sim

        loss.backward()
        optimizer.step()

        print(f"Step {step:4d} | Energy: {out_energy.mean().item():.4f} | CosSim: {avg_cos_sim.item():.4f} | KL: {kl_div.item():.4f}")

        if out_energy.mean().item() <= lim:
            os.makedirs('results', exist_ok=True)
            torch.save(model.state_dict(), 'results/degeneracy_model.pth')
            break
