import os
# os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import pennylane as qml
from itertools import product
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import scipy.io
import datetime

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
    for i in np.arange(0,nq-1):
        apply_2q_gate(input[idx_gate:idx_gate+5,:],wires=[i, (i+1)])
        idx_gate += 5

# Hamiltonian
def H_XXZ(nq, is_open):
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
    inputs = torch.reshape(inputs, shapes[0])
    qml.layer(layer_2U, nlayer, inputs)
    return qml.expval(H_XXZ(nq, False))  # False → PBC, True → OBC

@qml.qnode(dev, interface='auto')
def circuit_vec(inputs): # return final state to calculate fidelity
    qml.layer(layer_2U, nlayer, inputs)
    return qml.state()

if __name__ == '__main__':
    # # INFO
    # 18 qubits: -44.5364/18 = -2.4742

    # PARAMETERs
    nq = 18
    method = 'VQE' # SA, VQE
    nlayer = 48
    shapes = [(nlayer, 5*(nq-1), 3)]
    layer_1 = layer_2U
    learning_rate = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Files
    # 1. csv
    csv_name = 'vqe' + '_nq' + str(nq) + '_nl' + str(nlayer) + '_lr' + str(learning_rate)
    if method == 'SA':
        csv_name += '_init1e-2'
    csv_name = os.path.join(sys.path[0], "result/data", csv_name)

    # 2. ED state
    path = os.path.join(sys.path[0], "result", "GS" + "_" + str(nq) + ".mat")
    psi = torch.tensor(scipy.io.loadmat(path)['psi_T']).to(device)

    # # TRAINING
    n_repeat, n_loop = 10, 1000
    min_energy = 1e9
    vqe_grads = [] # (n_repeat * n_iterations, gradients)
    history = []
    fidelities = []

    for l in range(n_repeat):
        np.random.seed()
        # 1. initialization
        if method == 'SA':
            weights = [torch.tensor(np.random.random(size=shapes[0])*(1e-2), requires_grad=True)]
        elif method == 'VQE':
            weights = [torch.tensor(np.random.random(size=shapes[0])*2*np.pi, requires_grad=True)]

        history.append([])
        fidelities.append([])

        # 2. training
        opt = torch.optim.Adam(weights, lr=learning_rate)
        def closure():
            global vqe_grads_first, vqe_grads_middle, vqe_grads_last
            opt.zero_grad()
            loss = circuit(weights[0])
            loss.backward()
            grads = weights[0].grad
            vqe_grads.append(grads.detach().numpy().reshape(-1).tolist())
            return loss

        for epoch in range(n_loop):
            # updata parameters
            loss = closure()
            opt.step()

            # fidelity
            state = torch.reshape(circuit_vec(weights[0]), (-1,)).to(device)
            fidelity = qml.math.fidelity_statevector(state, psi, check_state=True).cpu().detach().numpy()

            # output/save data
            print('Repeat:', l, 'Epoch:', epoch, 'Energy:', loss.detach().numpy(), 'Fidelity:',  fidelity[0])
            min_energy = min(min_energy, loss)
            history[l].append(loss.detach().numpy())
            fidelities[l].append(fidelity)

        print('min_loss = ', min_energy)

        np.savetxt(csv_name + '_energy.csv', history, delimiter=',')
        np.savetxt(csv_name + '_fidelity.csv', fidelities, delimiter=',')

    np.savetxt(csv_name + '_grads.csv', vqe_grads, delimiter=',')


