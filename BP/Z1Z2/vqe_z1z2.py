import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2, 3"
import numpy as np
import torch
import pennylane as qml
import sys
import warnings
warnings.filterwarnings('ignore')

def H():
    coeffs = [1]
    obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
    return qml.Hamiltonian(coeffs, obs)

nq = 20
rot_gate = [qml.RX, qml.RY, qml.RZ]
dev = qml.device("lightning.gpu", wires=nq, batch_obs=True)
@qml.qnode(dev, interface='torch', diff_method='adjoint')
def circuit(weights, rot_idx):
    for l in range(nlayer):
        for i in range(nq):
            rot_gate[rot_idx[l, i]](weights[..., l, i], wires=i)

        if nq > 1:
            for i in range(nq):
                qml.CZ(wires=[i, (i+1)%nq])
    return qml.expval(H())

if __name__ == '__main__':
    # PARAMETERs
    # 1. quantum circuit
    nq = 20
    nlayer = 400
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 923
    np.random.seed(seed)
    rot_idx = np.random.choice([0, 1, 2], (nlayer, nq), replace=True) # for random rotation gates

    # 2. method selection @ files
    method = 'VQE' # 'VQE' or 'SA'
    file_name = 'vqe_nq%d_nl%d_seed%d' % (nq, nlayer, seed)
    if method == 'VQE':
        weights = [torch.tensor(np.random.random(size=(nlayer, nq)), requires_grad=True)]
        csv_name = os.path.join(sys.path[0], "result/data", file_name)
    elif method == 'SA':
        weights = [torch.tensor(np.random.random(size=(nlayer, nq))*0.01, requires_grad=True)]
        csv_name = os.path.join(sys.path[0], "result/data", file_name + '_init1e-2')

    # 3. training
    n_epoch = 300
    lr = 0.001
    opt = torch.optim.Adam(weights, lr=lr)

    # TRAINING
    vqe_grads = []
    def closure():
        opt.zero_grad()
        loss = circuit(weights[0], rot_idx)
        loss.backward()

        grads = weights[0].grad
        vqe_grads.append(grads.detach().numpy().reshape(-1).tolist())
        return loss

    history = []
    for epoch in range(n_epoch):
        loss = closure()
        history.append(loss.detach().numpy())
        opt.step()
        print('Epoch:', epoch, 'Energy:', loss.detach().numpy())

    print('min_loss = ', min(history))

    np.savetxt(csv_name + '_grads.csv', vqe_grads, delimiter=',')
    np.savetxt(csv_name + '_energy.csv', history, delimiter=',')