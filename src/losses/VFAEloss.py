# from https://github.com/yolomeus/DMLab2020VFAE/blob/master/losses.py
from torch import cat as torch_cat
from torch import zeros_like as torch_zeros_like
from torch import sum as torch_sum
from torch import min as torch_min
from torch import max as torch_max
from torch import tensor as torch_tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, Sigmoid

from .FastMMD import FastMMD

class VFAEloss(Module):
    def __init__(self, alpha=1.0, beta=1.0, mmd_dim=500, mmd_gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.bce = BCEWithLogitsLoss()
        self.ce = CrossEntropyLoss()
        self.mmd = FastMMD(mmd_dim, mmd_gamma)

    def forward(self, decoded, x, s, y):
        x_s = torch_cat([x, s], dim=-1)

        supervised_loss = self.ce(decoded['y_rec'], y.squeeze().long())
        if torch_min(x_s).item()<0 or torch_max(x_s).item()>1:
            reconstruction_loss = self.bce(decoded['x_rec'], Sigmoid()(x_s.float()))
        else:
            reconstruction_loss = self.bce(decoded['x_rec'], x_s.float())
        
        kl_loss_z1 = self._kl_gaussian(decoded['z1_enc_logvar'],
                                        decoded['z1_enc_mu'],
                                        decoded['z1_dec_logvar'],
                                        decoded['z1_dec_mu'])

        zeros = torch_zeros_like(decoded['z2_enc_mu'])
        kl_loss_z2 = self._kl_gaussian(decoded['z2_enc_logvar'],
                                        decoded['z2_enc_mu'],
                                        zeros,
                                        zeros)

        loss = reconstruction_loss + kl_loss_z1 + kl_loss_z2 + self.alpha * supervised_loss

        z1_enc = decoded['z1_encoded']
        z1_protected, z1_non_protected = self._separate_protected(z1_enc, s)
        if len(z1_protected) > 0 and len(z1_non_protected)>0:
            batch_size = x.shape[0]
            mmd = self.mmd(z1_protected, z1_non_protected)
            loss += self.beta * mmd * batch_size 
        else:
            mmd = torch_tensor([0])
        
        items = {
            'rec_loss': reconstruction_loss,
            'kl_loss_z1': kl_loss_z1,
            'kl_loss_z2': kl_loss_z2,
            'supervised_loss': supervised_loss,
            'mmd': mmd
        }

        return loss, items

    @staticmethod
    def _kl_gaussian(logvar_a, mu_a, logvar_b, mu_b):
        per_example_kl = logvar_b - logvar_a - 1 + (logvar_a.exp() + (mu_a - mu_b).square()) / logvar_b.exp()
        kl = 0.5 * torch_sum(per_example_kl, dim=1)
        return kl.mean()

    @staticmethod
    def _separate_protected(batch, s):
        idx_protected = (s == 0).nonzero()[:, 0]
        idx_non_protected = (s != 0).nonzero()[:, 0]
        protected = batch[idx_protected]
        non_protected = batch[idx_non_protected]

        return protected, non_protected