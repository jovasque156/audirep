from torch import cdist, nn
from torch import pow as torch_pow
from torch import mean as torch_mean
from torch.nn import MSELoss

# from https://github.com/rhythmswing/Fair-Representation-Learning/blob/master/model.py
class NRL(nn.Module): #n.Module was added
    def __init__(self, n_features=None, n_dim=None, n_protected=2, encoder=None, decoder=None, critic=None, device='cuda'):
        super().__init__()
        assert ((n_features is not None) and (n_dim is not None)) or (encoder is not None), '(n_features or n_dim) and encoder are None, only on can be None '
        assert ((n_features is not None) and (n_dim is not None)) or (decoder is not None), '(n_features or n_dim) and decoder are None, only on can be None'
        assert ((n_features is not None) and (n_dim is not None)) or (critic is not None), '(n_features or n_dim) and critic are None, only on can be None'

        self.n_prot = n_protected
        if encoder is None:
            self.encoder = nn.Sequential(nn.Linear(n_features, n_dim))
        else:
            self.encoder = encoder
        if decoder is None:
            self.decoder = nn.Sequential(nn.Linear(n_dim, n_features))
        else:
            self.decoder = decoder
        if critic is None:
            self.critic = [nn.Sequential(nn.Linear(n_dim,1))] * n_protected
        else:
            self.critic = critic

        if device=='cuda':
            self.cuda()
        else:
            self.cpu()

        self.criteria_dec = MSELoss(reduction='sum')

    def wdist(self, g, s, p):
        g_p = g[s.view(-1)==p]
        g_rest = g[s.view(-1)!=p]

        c_p = self.critic[p](g_p)
        c_rest = self.critic[p](g_rest)
        w_dist = cdist(c_p, c_rest, p=1)
        w_dist_count = w_dist.shape[0]
        
        return w_dist.sum()/w_dist_count, w_dist_count

    def cuda(self):
        self.encoder = self.encoder.cuda()
        self.decoder = self.decoder.cuda()
        for t in range(len(self.critic)):
            self.critic[t] = self.critic[t].cuda()

    def cpu(self):
        self.encoder.cpu()
        self.decoder.cpu()
        for t in range(len(self.critic)):
            self.critic[t] = self.critic[t].cpu()

    def forward(self, X, s, p):
        g = self.encoder(X)
        
        rec = self.decoder(g)
        mse_mean = torch_mean(torch_pow(rec-X,2))
        
        w_dist_mean, w_dist_counts = self.wdist(g, s, p)
        
        return mse_mean, rec.shape[0], w_dist_mean, w_dist_counts