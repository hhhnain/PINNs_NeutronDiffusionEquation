import torch
import torch.nn as nn

class Param_Encoder(nn.Module):
    def __init__(self):
        super(Param_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )

    def forward(self, ea, d):
        return self.encoder(torch.cat((ea, d), dim=1))

class Coordinates_Encoder(nn.Module):
    def __init__(self):
        super(Coordinates_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.encoder(torch.cat(x, dim=1))

class PINNs(nn.Module):
    def __init__(self):
        super(PINNs, self).__init__()
        self.encoder = Param_Encoder()
        self.register_buffer('lb_xy', torch.tensor([0.0]))
        self.register_buffer('ub_xy', torch.tensor([5.0]))
        self.register_buffer('lb_ea', torch.tensor([0.01]))
        self.register_buffer('ub_ea', torch.tensor([0.03]))
        self.register_buffer('lb_d', torch.tensor([0.39]))
        self.register_buffer('ub_d', torch.tensor([0.79]))


        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def normalise_param(self, X, lower_bounds, upper_bounds):
        return (X - lower_bounds) / (upper_bounds - lower_bounds)

    def normalise_coord(self, X, lower_bounds, upper_bounds):
        return 2.0 * (X - lower_bounds) / (upper_bounds - lower_bounds) - 1.0


    def forward(self, x, Ea, D):
        x_norm = self.normalise_coord(x, self.lb_xy, self.ub_xy)
        Ea_norm = self.normalise_param(Ea, self.lb_ea, self.ub_ea)
        D_norm = self.normalise_param(D, self.lb_d, self.ub_d)
        special_param = self.encoder(Ea_norm, D_norm)

        X = torch.cat((x_norm, special_param), dim=1)

        return self.decoder(X)

