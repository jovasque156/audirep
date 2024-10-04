import torch
from torchvision import transforms

def get_transform(method, image_size, reprogram_size=None, ):
    if method in ["std", "adv", "rpatch", "roptim"]:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
        ])
    elif method == "repro":
        assert reprogram_size is not None
        l_pad = int((image_size - reprogram_size + 1) / 2)
        r_pad = int((image_size - reprogram_size) / 2)

        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.Resize(reprogram_size),
            transforms.Pad(padding=(l_pad, l_pad, r_pad, r_pad)),
            transforms.RandomHorizontalFlip(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.Resize(reprogram_size),
            transforms.Pad(padding=(l_pad, l_pad, r_pad, r_pad)),
        ])
    else:
        raise ValueError

    return transform_train, transform_test

def get_one_hot(y, num_class, device):
    if len(y.shape) == 1:
        y_new = y.unsqueeze(-1)
    else:
        y_new = y
    y_one_hot = torch.FloatTensor(y_new.shape[0], num_class).to(device)
    y_one_hot.zero_()
    y_one_hot.scatter_(1, y_new, 1)
    return y_one_hot

import numpy as np

def demP(inputs, positive=1):
    pred = inputs['pred']
    sens = inputs['sens']
    
    groups = np.unique(sens)

    positive_rates = {}
    for g in groups:
        positive_rates[g] = np.mean(pred[sens == g] == positive)

    return np.max(list(positive_rates.values())) - np.min(list(positive_rates.values()))

def opp(inputs, positive=1):
    pred = inputs['pred']
    target = inputs['target']
    sens = inputs['sens']
    
    groups = np.unique(sens)

    true_pos_rates = {}
    for g in groups:
        true_pos_rates[g] = np.mean(pred[(sens == g) & (target == positive)] == positive)

    return np.max(list(true_pos_rates.values())) - np.min(list(true_pos_rates.values()))

def odd(inputs, positive=1):
    pred = inputs['pred']
    target = inputs['target']
    sens = inputs['sens']
    
    groups = np.unique(sens)

    odd_rates = {}
    for g in groups:
        odd_rates[g] = (np.mean(pred[(sens == g) & (target == positive)] == positive) + np.mean(pred[(sens == g) & (target != positive)] == positive))/2

    return np.max(list(odd_rates.values())) - np.min(list(odd_rates.values()))

def accuracy(inputs):
    return np.mean(inputs['pred'] == inputs['target'])

def f1score(inputs, positive=1):
    recall = sum((inputs['pred']==positive) & (inputs['target']==positive))/sum(inputs['target']==positive)
    precision = sum((inputs['pred']==positive) & (inputs['target']==positive))/sum(inputs['pred']==positive)
    
    return 2*(recall*precision)/(recall+precision)