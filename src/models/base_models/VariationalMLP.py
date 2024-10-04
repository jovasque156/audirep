#from https://github.com/yolomeus/DMLab2020VFAE/blob/master/model/vfae.py
from torch import nn as torch_nn
from torch import randn_like as torch_randn_like
from torch import exp as torch_exp
from torch.nn import LayerNorm
from .MLP import MLP

class VariationalMLP(torch_nn.Module):
    def __init__(self, in_features, hidden_dims, z_dim, activation_hidden, p_dropout=None):
        super().__init__()
        
        layers = []
        
        if hidden_dims:
            shapes = [in_features]+hidden_dims if isinstance(hidden_dims,list) else [in_features]+[hidden_dims]
        
            for i, o in zip(shapes[:-1], shapes[1:]):
                m = torch_nn.Linear(i, o)
                layers.append(m)
                if activation_hidden is not None:
                    if activation_hidden=='relu':
                        layers.append(torch_nn.ReLU())
                    if activation_hidden=='sigmoid':
                        layers.append(torch_nn.Sigmoid())
                    if activation_hidden=='softmax':
                        layers.append(torch_nn.Softmax(dim=1))
                    if activation_hidden=='leaky-relu':
                        layers.append(torch_nn.LeakyReLU(0.01))
                if p_dropout is not None:
                    layers.append(torch_nn.Dropout(p_dropout))
                bn = LayerNorm(o)
                layers.append(bn)
        else:
            shapes = [in_features]

        self.encoder = torch_nn.Sequential(*layers)
        
        if len(shapes)==1:
            self.logvar_encoder = torch_nn.Sequential(
                                                    LayerNorm(shapes[-1]),
                                                    torch_nn.Linear(shapes[-1], z_dim)
                                                    )
            self.mu_encoder = torch_nn.Sequential(
                                                    LayerNorm(shapes[-1]),
                                                    torch_nn.Linear(shapes[-1], z_dim)
                                                    )
        else:
            self.logvar_encoder = torch_nn.Linear(shapes[-1], z_dim)
            self.mu_encoder = torch_nn.Linear(shapes[-1], z_dim)

    def forward(self, inputs):
        x = self.encoder(inputs) if self.encoder else inputs

        mu = self.mu_encoder(x)
        logvar = (0.5 * torch_exp(self.logvar_encoder(x)))

        # reparameterization trick: we draw a random z
        epsilon = torch_randn_like(mu)
        z = epsilon * mu + logvar

        return z, logvar, mu