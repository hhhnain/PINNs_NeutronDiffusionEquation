import numpy as np
import torch
from model  import PINNs

# Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Model paths
model_path = 'vacuum_vacuum.pth'

NN = PINNs().to(device)

NN.load_state_dict(torch.load(model_path, map_location=device))
NN.eval()


# Material data
material_properties = np.loadtxt('../one-dimensional_cmfd/data.txt', skiprows=1)
sigmaA = material_properties[:, 0]
sigmaT = material_properties[:, 1]

def get_D(sigmaT):
    return 1/(3*sigmaT)

Diffusion_coefficient = get_D(sigmaT)

files = len(material_properties)

# Data loader
def data_loader(x_data, grad=True):
    if isinstance(x_data, np.ndarray):
        tensor = torch.from_numpy(x_data).float().to(device)
    elif isinstance(x_data, torch.Tensor):
        tensor = x_data.clone().detach().to(device)
    else:
        raise TypeError("x_data must be a NumPy array or a PyTorch tensor")
    tensor.requires_grad_(grad)
    return tensor.unsqueeze(-1)

# Prediction grid
x_p = np.linspace(0, 5, 100)
y_p = np.linspace(0, 5, 100)
x_pred = data_loader(x_p.flatten(), grad=False)
y_pred = data_loader(y_p.flatten(), grad=False)

# Initialize dictionaries to store errors
errors = {"rmse": [], "mean": [], "max": []}

# Loop over materials and models
for i in range(files):
    cmfd_results = np.load(f'../one-dimensional_cmfd/cmfd_{i}_flux.npy')[50, :]
    cmfd_flux = cmfd_results / np.mean(cmfd_results)

    ea = data_loader(np.array([sigmaA[i]]), False)
    d = data_loader(np.array([Diffusion_coefficient[i]]), False)

    ea = torch.ones_like(x_pred) * ea
    d = torch.ones_like(y_pred) * d

    # Loop through each model and compute errors
    with torch.no_grad():
        predictions = NN(x_pred, ea, d)

        flux = predictions.cpu().numpy()
        pinns_flux = flux / np.mean(flux)

        rel_err = np.abs(pinns_flux - cmfd_flux) #/ pinns_flux * 100
        errors["mean"].append(rel_err.mean())
        errors["max"].append(rel_err.max())

        rmse = np.sqrt(np.mean((pinns_flux - cmfd_flux) ** 2))
        errors["rmse"].append(rmse)


arr = np.array([errors["rmse"], errors["mean"], errors["max"]]).T
np.savetxt("all_errors.txt", arr, header="RMSE\tMeanError(%)\tMaxError(%)", fmt="%.6f")

print("✅ All model errors saved successfully!")



