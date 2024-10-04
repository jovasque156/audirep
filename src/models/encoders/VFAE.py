# From https://github.com/yolomeus/DMLab2020VFAE/blob/master/model/vfae.py
import torch
from torch.nn import Module, Dropout


class VFAE(Module):
    def __init__(self,
                encoder_z1,
                encoder_z2,
                decoder_z1,
                decoder_y,
                decoder_x,
                dropout_rate=0.0):
        super().__init__()
        self.encoder_z1 = encoder_z1
        self.encoder_z2 = encoder_z2

        self.decoder_z1 = decoder_z1
        
        self.decoder_y = decoder_y
        self.decoder_x = decoder_x

        self.dropout = Dropout(dropout_rate)

    def forward(self, x,s,y):
        # encode
        x_s = torch.cat([x, s], dim=1)
        x_s = self.dropout(x_s)
        z1_encoded, z1_enc_logvar, z1_enc_mu = self.encoder_z1(x_s)

        z1_y = torch.cat([z1_encoded, y], dim=1)
        z2_encoded, z2_enc_logvar, z2_enc_mu = self.encoder_z2(z1_y)

        # decode
        z2_y = torch.cat([z2_encoded, y], dim=1)
        z1_decoded, z1_dec_logvar, z1_dec_mu = self.decoder_z1(z2_y)

        z1_s = torch.cat([z1_decoded, s], dim=1)
        # Or reconstructed
        x_rec = self.decoder_x(z1_s)
        y_rec = self.decoder_y(z1_encoded)

        
        outputs = {
            # predictive outputs
            'x_rec': x_rec,
            'y_rec': y_rec,
            'z1_encoded': z1_encoded,

            # outputs for regularization loss terms
            'z1_enc_logvar': z1_enc_logvar,
            'z1_enc_mu': z1_enc_mu,

            'z2_enc_logvar': z2_enc_logvar,
            'z2_enc_mu': z2_enc_mu,
            
            #this are giving nans
            'z1_dec_logvar': z1_dec_logvar,
            'z1_dec_mu': z1_dec_mu
        }

        return outputs

