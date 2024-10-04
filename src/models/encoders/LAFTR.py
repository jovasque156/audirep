import torch
import torch.nn as nn
import numpy as np

# from https://github.com/ahxt/fair_fairness_benchmark/blob/master/src/networks.py#L62C1-L80C35
class RMSELoss(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class LAFTRLoss(nn.Module):
    def __init__(self):
        super().__init__(A_weights, Y_weights, AY_weights)

        self.A_weights = None
        self.Y_weights = None
        self.AY_weights = None

    def DP(s_pred, s_true):
        pass

    def EO(s_pred, s_true, y_true):
        pass

def get_A_proportions(s_train):
    distr = s_train.iloc[:,0].value_counts().values
    return [1/d for d in distr]

def get_Y_proportions(y_train):
    distr = y_train.iloc[:,0].value_counts().values
    return [1/d for d in distr]

def get_AY_proportions(s_train,y_train):
    arr = np.concatenate((s_train.values, y_train.values), axis=1)
    _, counts = np.unique(arr, axis=0, return_counts=True)
    return [[1/x, 1/y] for x, y in zip(counts[::2], counts[1::2])]

def get_adv_loss(s_train, y_train):
    A_weights = get_A_proportions(s_train)
    Y_weights = get_A_proportions(y_train)
    AY_weights = get_AY_proportions(s_train, y_train)
    return LAFTRLoss(A_weights, Y_weights, AY_weights)

class LAFTR(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        adversary,
        classifier,
        rec_loss=None,
        adv_loss=None,
        classif_loss=None,
        A_x=1,
        A_y=1,
        A_z=50,
        device='cuda'
    ):
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.adversary = adversary.to(device)
        self.classifier = classifier.to(device)
        if rec_loss is None:
            rec_loss = RMSELoss()
        self.rec_loss = rec_loss
        
        if adv_loss is None:
            adv_loss = nn.CrossEntropyLoss()
        self.adv_loss = adv_loss
        
        if classif_loss is None:
            classif_loss = nn.CrossEntropyLoss()
            
        self.classif_loss = classif_loss
        self.A_x = A_x
        self.A_y = A_y
        self.A_z = A_z

        self.A_weights = None
        self.Y_weights = None
        self.AY_weights = None

    def forward(self, x, is_protected, y):
        if len(is_protected.shape) == 1:
            is_protected = is_protected[:, None]
        if len(y.shape) == 1:
            y = y[:, None]
        
        encoded = self.encoder(x)      
        decoded = self.decoder(encoded)        
        if self.classifier is not None:
            classif_logits = self.classifier(encoded)
        else:
            classif_logits = None
        adv_logits = self.adversary(encoded)

        return encoded, decoded, classif_logits, adv_logits

    def loss(self,x, is_protected, y, decoded, classif_logits, adv_logits):
        if len(is_protected.shape) == 1:
            is_protected = is_protected[:, None]
        if len(y.shape) == 1:
            y = y[:, None]

        L_x = self.rec_loss(x, decoded)

        L_y = (
            self.classif_loss(classif_logits, y.view(-1).long())
            if classif_logits is not None
            else torch.tensor(0.0)
        )

        L_z = self.adv_loss(adv_logits, is_protected.view(-1).long())
        
        return self.A_x * L_x + self.A_y * L_y - self.A_z * L_z, L_x, L_y, L_z

    def fair_parameters(self):
        for m in [self.encoder, self.decoder, self.classifier]:
            if m is not None:
                yield from m.parameters()

    def adv_parameters(self):
        yield from self.adversary.parameters()
