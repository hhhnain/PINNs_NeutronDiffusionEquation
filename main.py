import torch
import numpy as np
from model import PINNs

def data_loader(x_data, grad=True, dev='mps'):
    tensor = torch.tensor(x_data, requires_grad=grad, dtype=torch.float32, device=dev)
    return tensor.unsqueeze(1)


def grad_a(a, b, graph=True):
    da_db = torch.autograd.grad(a.sum(), b, create_graph=graph)[0]
    return da_db

ea_max = 0.03
ea_min = 0.01
es_max = 0.84
es_min = 0.40
Q = 0.1


def generate_parameter_pairs(num_pairs):
    """Generate sigmaA and D pairs using the physical relationship D = 1/(3*(sigmaS + sigmaA))"""
    # Sample absorption cross-sections
    sigmaA_vals = np.random.uniform(ea_min, ea_max, num_pairs)

    # Sample scattering cross-sections
    sigmaS_vals = np.random.uniform(es_min, es_max, num_pairs)

    # Calculate diffusion coefficients using physical relationship
    sigmaTotal_vals = sigmaS_vals + sigmaA_vals
    D_vals = 1.0 / (3.0 * sigmaTotal_vals)

    return sigmaA_vals, D_vals


NN = PINNs().to(device='mps')

# Try loading previous checkpoint
checkpoint_path = "vacuum_vacuum.pth"
try:
    NN.load_state_dict(torch.load(checkpoint_path, map_location='mps'))
    print(f"Loaded checkpoint from {checkpoint_path}")
except FileNotFoundError:
    print("No checkpoint found, starting training from scratch")

optimizer = torch.optim.Adam(NN.parameters(), lr=1e-4)
x_phy = np.linspace(0., 5., 10000)
x_phy = data_loader(x_phy)

epochs = 100000
plot_steps = 500
batch_size = 1

for i in range(epochs):
    optimizer.zero_grad()

    sigmaA_vals, D_vals = generate_parameter_pairs(batch_size)

    sigmaA = data_loader(sigmaA_vals, True).expand_as(x_phy)
    D_approx = data_loader(D_vals, True).expand_as(x_phy)

    phi = NN(x_phy, sigmaA, D_approx)

    dphi_dx = torch.autograd.grad(phi.sum(), x_phy, create_graph=True)[0]
    dphi2_dx2 = torch.autograd.grad(dphi_dx.sum(), x_phy, create_graph=True)[0]

    pde = -D_approx*dphi2_dx2 + sigmaA*phi - Q

    Physics_loss = torch.mean(pde ** 2)


    left_eqn = D_approx[0] * dphi_dx[0] - 0.5 * phi[0]
    right_eqn = -D_approx[-1] * dphi_dx[-1] - 0.5 * phi[-1]

    bc_loss = torch.mean(left_eqn**2) + torch.mean(right_eqn**2)

    loss = Physics_loss + bc_loss
    loss.backward()
    optimizer.step()

    if i % plot_steps == 0:
        print(f'Epochs [{i}/{epochs + 1}], Total Loss: {loss.item():.8f}, Physics_loss: {Physics_loss.item():.8f}, bc_loss: {bc_loss.item():.8f}')
        torch.save(NN.state_dict(), 'vacuum_vacuum.pth')


