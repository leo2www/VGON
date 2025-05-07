import os
# os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import pennylane as qml
import sys
import warnings
warnings.filterwarnings('ignore')

# Circuit ansatz
nq = 20
nlayer = 400
dev = qml.device("lightning.gpu", wires=nq, batch_obs=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

global rot_idx
rot_idx = None
rot_gate = [qml.RX, qml.RY, qml.RZ]
def layer_random(inputs, rot_idx):
    for i in range(nq):
        rot_gate[rot_idx[i]](inputs[i], wires=i)

    if nq > 1:
        for i in range(nq):
            qml.CZ(wires=[i, (i+1)%nq])

dev = qml.device("lightning.gpu", wires=nq, batch_obs=False)
@qml.qnode(dev, interface='torch', diff_method='adjoint')
def circuit(inputs):
    qml.layer(layer_random, nlayer, inputs, rot_idx)
    return qml.expval(H())  # False → PBC, True → OBC

# Hamiltonian
def H():
    coeffs = [1]
    obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
    return qml.Hamiltonian(coeffs, obs)

# VGON model
class Model(nn.Module):
    def __init__(self, para_dim, z_dim, h_dim):
        super(Model, self).__init__()
        # encoder
        self.e1 = nn.ModuleList([nn.Linear(para_dim, h_dim[0])])
        self.e1 += [nn.Linear(h_dim[i-1], h_dim[i]) for i in range(1, len(h_dim))]
        self.e2 = nn.Linear(h_dim[-1], z_dim) # get mean prediction
        self.e3 = nn.Linear(h_dim[-1], z_dim) # get mean prediction

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
        log_var = self.e3(h)    # log of variance, easy for computing
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
        return circuit(out.transpose(0, 1).reshape((nlayer, nq, z.shape[0])))

    def forward(self, x):
        mean, log_var = self.encoder(x)
        # trick: reparameterization trick, sampling
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var
    

if __name__ == '__main__':
    # PARAMETERs
    # 1. quantum circuit
    nq, nlayer = 20, 400
    para_dim = np.prod((nq, nlayer))
    seed = 923
    np.random.seed(seed)
    rot_idx = np.random.choice([0, 1, 2], (nlayer, nq), replace=True) # for random rotation gates

    # 2. file
    file_name = 'vgon_nq%d_nl%d_seed%d' % (nq, nlayer, seed)
    csv_name = os.path.join(sys.path[0], "result/data", file_name)
    model_name = os.path.join(sys.path[0], "result/model", file_name)

    # 3. VGON
    z_dim = 3
    h_dim = [256, 128, 64, 32]
    lr = 0.0001
    batch_size = 4
    epochs = 1
    max_iter = 300

    x = torch.rand(10000, para_dim) * 2 * torch.pi
    train_db = torch.utils.data.DataLoader(
        list(x), shuffle=True, batch_size=batch_size, drop_last=True
    )
    
    model = Model(para_dim, z_dim, h_dim)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(model)

    # TRAINING
    min_loss = 1e9
    vgon_grads = []
    vgon_history = []
    for epoch in range(epochs):
        running_loss = 0
        for i, batch_x in enumerate(train_db):
            batch_x = batch_x.to(device)
            opt.zero_grad()
            out, mean, log_var = model(batch_x)
            energy = torch.mean(out)
            kl_div = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
            kl_div = kl_div.mean() / out.shape[0]
            loss_evaluated = energy + 0.5 * kl_div
            loss_evaluated.backward()

            vgon_grads.append(model.d5.bias.grad.cpu().detach().numpy().reshape(-1).tolist())

            opt.step()
            running_loss += loss_evaluated
            vgon_history.append([torch.min(out).cpu().detach().numpy(), energy.cpu().detach().numpy(), kl_div.cpu().detach().numpy()])
            print('Iteration', i, 'loss =', str(loss_evaluated.cpu().detach().numpy()), 'mean energy =', str(vgon_history[-1][1]), 'kl_div =', str(vgon_history[-1][2]))

            if min_loss > loss_evaluated:
                min_loss = loss_evaluated
                torch.save(model.state_dict(), model_name + '.pth')
            
            if i >= max_iter:
                break

    print('min_loss =', min_loss)

    np.savetxt(csv_name + '_grads.csv', vgon_grads, delimiter=',')
    np.savetxt(csv_name + '_history.csv', vgon_history, delimiter=',')
            