# Utilites
import os
import argparse
import pandas as pd
from numpy import array
from pandas import concat as pd_concat
from utils import demP, opp, odd
from sklearn.preprocessing import StandardScaler

# Load configs
from configs.config_datasets import data_stats
from configs.config_nrl import config_NRL
from configs.config_laftr import config_LAFTR
from configs.config_vfae import config_VFAE
from configs.config_classifiers import config_classifiers

# Import internal libraries
from src.models.encoders.NRL import NRL
from src.models.encoders.VFAE import VFAE
from src.models.encoders.LAFTR import LAFTR
from src.models.base_models.MLP import MLP
from src.models.base_models.VariationalMLP import VariationalMLP
from src.models.data_loader.PandasDataSet import PandasDataSet_WithEncoder

# Torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import argmax as torch_argmax
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Softmax
from torch import log2 as torch_log2
from torch import load as torch_load
from torch import save as torch_save
from torch import gather as torch_gather
from torch.optim import Adam, SGD, AdamW
from torch import manual_seed
from torch import ones_like
from torch.cuda import is_available as cuda_is_available

import ipdb

DEVICE = 'cuda' if cuda_is_available() else 'cpu'

def total_epochs(t_0, t_mult, num_cycles):
    if t_mult>1:
        return int(t_0*(1-t_mult**num_cycles)/(1-t_mult))
    else:
        return t_0 * num_cycles

def load_vfae(args, epoch=349):
    manual_seed(0)
    
    checkpoint_path = f'results/vfae/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_encoder}/train_epoch_{epoch}_model.pt'
    if os.path.isfile(checkpoint_path):
        checkpoint_train = torch_load(checkpoint_path)
    else:
        checkpoint_path = f'results/vfae/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_encoder}/train_epoch_199_model.pt'
        checkpoint_train = torch_load(checkpoint_path)

    encoder_z1 = VariationalMLP(in_features= data_stats[args.dataset]['n_features']+data_stats[args.dataset]['s_dim'],
                            hidden_dims= config_VFAE[args.dataset][args.id_setting_encoder]['z1_enc_dim'],
                            z_dim= config_VFAE[args.dataset][args.id_setting_encoder]['z_dim'],
                            activation_hidden = config_VFAE[args.dataset][args.id_setting_encoder]['z1_enc_activation_hidden'],
                            p_dropout = config_VFAE[args.dataset][args.id_setting_encoder]['z1_enc_p_dropout'],
                        )
    encoder_z2 = VariationalMLP(
                                in_features= config_VFAE[args.dataset][args.id_setting_encoder]['z_dim']+data_stats[args.dataset]['y_dim'],
                                hidden_dims= config_VFAE[args.dataset][args.id_setting_encoder]['z2_enc_dim'],
                                z_dim= config_VFAE[args.dataset][args.id_setting_encoder]['z_dim'],
                                activation_hidden = config_VFAE[args.dataset][args.id_setting_encoder]['z2_enc_activation_hidden'],
                                p_dropout = config_VFAE[args.dataset][args.id_setting_encoder]['z2_enc_p_dropout'],
                                )
    decoder_z1 = VariationalMLP(
                                in_features= config_VFAE[args.dataset][args.id_setting_encoder]['z_dim']+data_stats[args.dataset]['y_dim'],
                                hidden_dims= config_VFAE[args.dataset][args.id_setting_encoder]['z1_dec_dim'],
                                z_dim= config_VFAE[args.dataset][args.id_setting_encoder]['z_dim'],
                                activation_hidden = config_VFAE[args.dataset][args.id_setting_encoder]['z1_dec_activation_hidden'],
                                p_dropout = config_VFAE[args.dataset][args.id_setting_encoder]['z1_enc_p_dropout'],
                                )
    y_out_dim = 2 if data_stats[args.dataset]['y_dim']==1 else data_stats[args.dataset]['y_dim']
    decoder_y = MLP(input_size=config_VFAE[args.dataset][args.id_setting_encoder]['z_dim'],
                    hidden_layers=config_VFAE[args.dataset][args.id_setting_encoder]['x_dec_dim'],
                    output_size=y_out_dim,
                    activation_hidden=config_VFAE[args.dataset][args.id_setting_encoder]['x_dec_activation_hidden'],
                    activation_output=config_VFAE[args.dataset][args.id_setting_encoder]['x_dec_activation_output'],
                    p_dropout=config_VFAE[args.dataset][args.id_setting_encoder]['x_dec_p_dropout'])
    #Free memory
    y_out_dim=None

    decoder_x = MLP(input_size=config_VFAE[args.dataset][args.id_setting_encoder]['z_dim']+data_stats[args.dataset]['s_dim'],
                    hidden_layers=config_VFAE[args.dataset][args.id_setting_encoder]['x_dec_dim'],
                    output_size=data_stats[args.dataset]['n_features']+data_stats[args.dataset]['s_dim'],
                    activation_hidden=config_VFAE[args.dataset][args.id_setting_encoder]['x_dec_activation_hidden'],
                    activation_output=config_VFAE[args.dataset][args.id_setting_encoder]['x_dec_activation_output'],
                    p_dropout=config_VFAE[args.dataset][args.id_setting_encoder]['x_dec_p_dropout'])

    vfae_model = VFAE(
                encoder_z1=encoder_z1,
                encoder_z2=encoder_z2,
                decoder_z1=decoder_z1,
                decoder_y=decoder_y,
                decoder_x=decoder_x).to(DEVICE)

    vfae_model.load_state_dict(checkpoint_train['vfae_state_dict'])

    return vfae_model.encoder_z1, checkpoint_train['parameters']['batch_size']

def load_nrl(args, epoch=349):
    manual_seed(0)
    checkpoint_train = torch_load(f'results/nrl/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_encoder}/train_epoch_{epoch}_model.pt')
    encoder = MLP(
                    input_size = data_stats[args.dataset]['n_features'], 
                    hidden_layers=config_NRL[args.dataset][args.id_setting_encoder]['encoder_hdims'],
                    output_size=config_NRL[args.dataset][args.id_setting_encoder]['z_dim'],
                    activation_hidden=config_NRL[args.dataset][args.id_setting_encoder]['encoder_activation_hidden'],
                    activation_output=config_NRL[args.dataset][args.id_setting_encoder]['encoder_activation_output']
                )
    decoder = MLP(
                input_size = config_NRL[args.dataset][args.id_setting_encoder]['z_dim'],
                hidden_layers = config_NRL[args.dataset][args.id_setting_encoder]['decoder_hdims'],
                output_size=data_stats[args.dataset]['n_features'],
                activation_hidden=config_NRL[args.dataset][args.id_setting_encoder]['decoder_activation_hidden'],
                activation_output=config_NRL[args.dataset][args.id_setting_encoder]['decoder_activation_output'] 
            )

    if data_stats[args.dataset]['n_protected']>2:
        critic = [
                MLP(
                    input_size = config_NRL[args.dataset][args.id_setting_encoder]['z_dim'],
                    hidden_layers = config_NRL[args.dataset][args.id_setting_encoder]['critic_hdims'],
                    output_size=1,
                    activation_hidden=config_NRL[args.dataset][args.id_setting_encoder]['critic_activation_hidden'],
                    activation_output=config_NRL[args.dataset][args.id_setting_encoder]['critic_activation_output'] 
                )
            ]*data_stats[args.dataset]['n_protected']
    else:
        critic = [
                MLP(
                    input_size = config_NRL[args.dataset][args.id_setting_encoder]['z_dim'],
                    hidden_layers = config_NRL[args.dataset][args.id_setting_encoder]['critic_hdims'],
                    output_size=1,
                    activation_hidden=config_NRL[args.dataset][args.id_setting_encoder]['critic_activation_hidden'],
                    activation_output=config_NRL[args.dataset][args.id_setting_encoder]['critic_activation_output'] 
                )
            ]

    nrl_model = NRL(
                    encoder = encoder,
                    decoder = decoder,
                    critic = critic,
                    device=DEVICE
                ).to(DEVICE)
    
    nrl_model.load_state_dict(checkpoint_train['nrl_state_dict'])

    return nrl_model.encoder, checkpoint_train['parameters']['batch_size']

def load_laftr(args, epoch=149):
    manual_seed(0)
    checkpoint_train = torch_load(f'results/laftr/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_encoder}/train_epoch_{epoch}_model.pt')
    
    # Loading model settings
    encoder = MLP(
                input_size = data_stats[args.dataset]['n_features'],
                hidden_layers = config_LAFTR[args.dataset][args.id_setting_encoder]['encoder_hdims'], 
                output_size = config_LAFTR[args.dataset][args.id_setting_encoder]['z_dim'], 
                activation_hidden = config_LAFTR[args.dataset][args.id_setting_encoder]['encoder_activation_hidden'], 
                activation_output = config_LAFTR[args.dataset][args.id_setting_encoder]['encoder_activation_output'], 
                p_dropout=config_LAFTR[args.dataset][args.id_setting_encoder]['encoder_p_dropout'])
                
    decoder = MLP(
                input_size = config_LAFTR[args.dataset][args.id_setting_encoder]['z_dim'],
                hidden_layers = config_LAFTR[args.dataset][args.id_setting_encoder]['decoder_hdims'], 
                output_size = data_stats[args.dataset]['n_features'], 
                activation_hidden = config_LAFTR[args.dataset][args.id_setting_encoder]['decoder_activation_hidden'], 
                activation_output = config_LAFTR[args.dataset][args.id_setting_encoder]['decoder_activation_output'], 
                p_dropout=config_LAFTR[args.dataset][args.id_setting_encoder]['decoder_p_dropout'])
                        
    adversary = MLP(
                input_size = config_LAFTR[args.dataset][args.id_setting_encoder]['z_dim'], 
                hidden_layers = config_LAFTR[args.dataset][args.id_setting_encoder]['adversary_hdims'], 
                output_size =  data_stats[args.dataset]['n_protected'], 
                activation_hidden = config_LAFTR[args.dataset][args.id_setting_encoder]['adversary_activation_hidden'], 
                activation_output = config_LAFTR[args.dataset][args.id_setting_encoder]['adversary_activation_output'], 
                p_dropout=config_LAFTR[args.dataset][args.id_setting_encoder]['adversary_p_dropout'])
                
    classifier = MLP(
                input_size = config_LAFTR[args.dataset][args.id_setting_encoder]['z_dim'], 
                hidden_layers = config_LAFTR[args.dataset][args.id_setting_encoder]['classifier_hdims'], 
                output_size = data_stats[args.dataset]['n_target'], 
                activation_hidden = config_LAFTR[args.dataset][args.id_setting_encoder]['classifier_activation_hidden'], 
                activation_output = config_LAFTR[args.dataset][args.id_setting_encoder]['classifier_activation_output'], 
                p_dropout=config_LAFTR[args.dataset][args.id_setting_encoder]['classifier_p_dropout'])

    laftr_model = LAFTR(
                    encoder = encoder,
                    decoder = decoder,
                    adversary = adversary,
                    classifier = classifier,
                    A_x=checkpoint_train['parameters']['coefs_A_x'],
                    A_y=checkpoint_train['parameters']['coefs_A_y'],
                    A_z=checkpoint_train['parameters']['coefs_A_z'],
                    device=DEVICE
                ).to(DEVICE)
    
    laftr_model.load_state_dict(checkpoint_train['laftr_state_dict'])

    return laftr_model.encoder, checkpoint_train['parameters']['batch_size']

def train_classifier(cl, opt, sch, train_l, val_l, test_l, path_base, args):
    manual_seed(0)
    loss_train = []
    loss_val = []
    loss_test = []
    
    acc_train = []
    acc_val = []
    acc_test = []

    if args.compute_pvi:
        pvi = []
    else:
        pvi = None

    if args.compute_disparities:
        eq_opp = []
        eq_odd = []
        dem_p = []
    else:
        eq_opp = None
        eq_odd = None
        dem_p = None

    ce = CrossEntropyLoss()

    steps_per_epoch = len(train_l)

    for epoch in range(args.n_epochs):
        loss_total = 0
        acc_total = 0
        total = 0
        cl.train()
        for i, (X,label,s) in enumerate(train_l):
            opt.zero_grad()

            X, label, s = X.to(DEVICE), label.to(DEVICE), s.to(DEVICE)
            label = 1*(label==0)
            
            loss = ce(cl(X), label)
            loss.backward()
            opt.step()
            if args.use_scheduler:
                sch.step(epoch + i/steps_per_epoch)

            loss_total += loss.item()*X.shape[0]
            total += X.shape[0]

            acc_total += sum(1*(torch_argmax(cl(X), dim=1)==label)).item()

        loss_train.append(loss_total/total)
        acc_train.append(acc_total/total)

        # Log the results so far
        if not args.debugging_mode:
            torch_save(
                {
                    'classifier_state_dict': cl.state_dict(),
                    'classifier_structure': str(cl),
                    'optimizer': opt.state_dict(),
                    'optimizer_structure': str(opt),
                    'scheduler': str(sch),
                    'parameters': vars(args)
                },
                f"{path_base}/train_epoch_{epoch}_model.pt"
            )

            torch_save(
                {
                    'total_loss': loss_train,
                    'total_acc': acc_train,
                    'parameters': vars(args)
                },
                f"{path_base}/train_epoch_results.pt"
            )

        loss_total = 0
        acc_total = 0
        total = 0
        cl.eval()
        for i, (X, label, s) in enumerate(val_l):
            X = X.to(DEVICE)
            label = label.to(DEVICE)
            s = s.to(DEVICE)
            
            label = 1*(label==0)

            loss_total += ce(cl(X),label).item()*X.shape[0]
            acc_total += sum(1*(torch_argmax(cl(X), dim=1)==label)).item()
            total += X.shape[0]

        loss_val.append(loss_total/total)
        acc_val.append(acc_total/total)

        if not args.debugging_mode:
            torch_save(
                {
                    'total_loss': loss_val,
                    'total_acc': acc_val,
                    'parameters': vars(args)
                },
                f"{path_base}/val_epoch_results.pt"
            )

        loss_total = 0
        acc_total = 0
        total = 0
        if args.compute_pvi:
            pvi_total = 0
        if args.compute_disparities:
            labels = []
            sensitives = []
            predictions = []

        distr = Softmax(dim=1)
        for i, (X,label,s) in enumerate(test_l):
            X, label, sen = X.to(DEVICE), label.to(DEVICE), s.to(DEVICE)
            label = 1*(label==0)
            
            loss_total += ce(cl(X), label).item()*X.shape[0]
            acc_total += sum(1*(torch_argmax(cl(X), dim=1)==label)).item()
            total += X.shape[0]

            if args.compute_pvi:
                pvi_total+=-1*torch_log2(torch_gather(distr(cl(X)),1,label.view(-1,1).long().detach())).sum().item()
            if args.compute_disparities:
                labels += label.detach().view(-1).tolist()
                sensitives += sen.detach().view(-1).tolist()
                predictions += torch_argmax(distr(cl(X)), dim=1).detach().view(-1).tolist()

        loss_test.append(loss_total/total)
        acc_test.append(acc_total/total)
        if args.compute_pvi:
            pvi.append(pvi_total/total)
        if args.compute_disparities:
            inputs = {
                'pred': array(predictions),
                'target': array(labels),
                'sens': array(sensitives)
            }
            eq_opp.append(opp(inputs))
            eq_odd.append(odd(inputs))
            dem_p.append(demP(inputs))

        if not args.debugging_mode:
            torch_save(
                {
                    'total_loss': loss_test,
                    'total_acc': acc_test,
                    'pvi': pvi,
                    'eq_opp': eq_opp,
                    'eq_odd': eq_odd,
                    'dem_p': dem_p,
                    'parameters': vars(args)
                },
                f"{path_base}/test_epoch_results.pt"
            )

        if args.debugging_mode:
            print(f'Epoch: {epoch+1}: acc {acc_test[-1]:.2f}, pvi {pvi[-1]:.2f}, eq_opp {eq_opp[-1] if eq_opp is not None else 0:.2f}, eq_odd {eq_odd[-1] if eq_odd is not None else 0:.2f}, dem_p {dem_p[-1] if dem_p is not None else 0:.2f}')

if __name__=='__main__':
    manual_seed(0)
    parser = argparse.ArgumentParser(description="AudiRep Project Experiments")
    
    # Setting for encoder
    parser.add_argument('--encoder', type=str, default=None, help='Encoder to be used (vfae,laftr,nrl)')
    parser.add_argument('--id_setting_encoder', type=str, default=None, help='Setting id of the trained encoder (from v0 to v9)')
    parser.add_argument('--lowest_loss', action='store_true', help='Set it to use the encoder state with lowest loss in the val set')

    # Setting for dataset
    parser.add_argument('--dataset', type=str, default='compas', help='Dataset to be used')
    parser.add_argument('--stratify_by', type=str, default='target+sensitive', help='Stratfy variable used')
    parser.add_argument('--scaler', type=str, default='MinMax', help='Scaler of the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for all dataloaders')

    # Setting for model
    parser.add_argument('--id_setting_model', type=str, default='v0', help='Setting id of the model to be trained (from v0 to v9)')
    parser.add_argument('--use_pretrained_classifier_target', action='store_true', help='Path to the classifiers to load and use its weights')
    # example: results/None/adult/standard_target+sensitive/classifiers/target_from_None/

    # Task
    parser.add_argument('--label', type=str, default='target', help='Define the label: target or sensitive')

    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='Adam', help='Set optimizer between (SGD, Adam, AdamW). Default: Adam')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for training classifier. Default: 5e-5')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--use_scheduler', action='store_true', help='Use for using scheduler CosineAnnealingWarmRestarts')
    parser.add_argument('--lr_max', type=float, default=5e-5, help='Set the max learning rate for scheduler')
    parser.add_argument('--lr_min', type=float, default=0.0, help='Set the min learning rate for scheduler')
    parser.add_argument('--T_0', type=int, default=10, help='Number of epochs for first cycle')
    parser.add_argument('--T_mult', type=int, default=1, help='Multiplier for cycles>1')
    parser.add_argument('--num_cycles', type=int, default=20, help='Number of cycles')

    # Setting for computing disparities
    parser.add_argument('--compute_disparities', action='store_true', help='Set to compute disparities from val and test dataset')
    parser.add_argument('--compute_pvi', action='store_true', help='Set to compute pvi from test dataset')

    # For debugging
    parser.add_argument('--debugging_mode', action='store_true', help='Use for avoid saving progress')
    args = parser.parse_args()

    print(f'Runing: {args}')

    # Check if id_setting_encoder is given when encoder
    if args.encoder not in ['constant', 'sensitive', 'target']:
        assert not(bool(args.encoder)!=bool(args.id_setting_encoder)), 'Encoder or Id setting of encoder was given, but not both'

    # Set epoch to use
    assert (not args.lowest_loss) or (args.encoder not in [None,'constant']), 'lowest_loss used but no encoder was given'
    if args.lowest_loss:
        print('Setting epoch to checkpoint with the lowest loss')
        checkpoint_val = torch_load(f"results/{args.encoder}/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_encoder}/val_epoch_results.pt")
        epoch = checkpoint_val['last_significant_best_epoch']
        checkpoint_val = None
    else:
        if args.encoder not in [None, 'constant','sensitive','target']:
            print('Setting epoch to checkpoint with the last loss')
            checkpoint_val = torch_load(f"results/{args.encoder}/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_encoder}/val_epoch_results.pt")
            epoch = len(checkpoint_val['loss_total'])-1
            checkpoint_val = None

    # Load encoder
    if args.encoder:
        print(f'Loading {args.encoder} {args.id_setting_encoder}')
        if args.encoder=='nrl':
            encoder, batch_size = load_nrl(args, epoch)
        elif args.encoder=='vfae':
            encoder, batch_size = load_vfae(args, epoch)
            pass
        elif args.encoder=='laftr':
            encoder, batch_size = load_laftr(args, epoch)
            pass
        elif args.encoder=='constant':
            encoder = ones_like
            batch_size = 254 if args.dataset=='compas' else 512
        elif args.encoder in ['target', 'sensitive']:
            encoder = None
        else:
            raise ValueError("Encoder not supported")
    else:
        encoder = None
        batch_size = 254 if args.dataset=='compas' else 512

    # Reset batch size if given
    batch_size = args.batch_size if args.batch_size else batch_size
    
    # Load datasets
    print(f'Loading dataset {args.dataset}, with scaler {args.scaler} and stratify by {args.stratify_by}')
    checkpoint = torch_load(f'results/datasets/{args.dataset}_stratifyby_{args.stratify_by}_scaler_{args.scaler}.pt')

    train_data, val_data, test_data = checkpoint['train_data'], checkpoint['val_data'], checkpoint['test_data']

    X_train, y_train, s_train = train_data
    X_val, y_val, s_val = val_data
    X_test, y_test, s_test = test_data
    
    if args.label=='target':
        label_train, label_val, label_test = y_train, y_val, y_test
        output_size = data_stats[args.dataset]['n_target']
    elif args.label=='sensitive':
        label_train, label_val, label_test = s_train, s_val, s_test
        output_size = 4 if args.dataset=='compas' else data_stats[args.dataset]['n_protected']
    elif args.label=='both':
        y_train = y_train*4 if args.dataset=='compas' else y_train*data_stats[args.dataset]['n_protected']
        y_val = y_val*4 if args.dataset=='compas' else y_val*data_stats[args.dataset]['n_protected']
        y_test = y_test*4 if args.dataset=='compas' else y_test*data_stats[args.dataset]['n_protected']
        label_train, label_val, label_test = pd.DataFrame(y_train.values+s_train.values, index=y_train.index), pd.DataFrame(y_val.values+s_val.values, index=y_val.index), pd.DataFrame(y_test.values+s_test.values, index=y_test.index)
        output_size = data_stats[args.dataset]['n_protected']*data_stats[args.dataset]['n_target']

    if args.encoder:
        if args.encoder=='vfae':
            X_train = pd_concat((X_train, s_train), axis=1)
            X_val = pd_concat((X_val, s_val), axis=1)
            X_test = pd_concat((X_test, s_test), axis=1)
        elif args.encoder=='sensitive':
            assert args.label=='target', 'if args.encoder is sensitive, thus args.label must be target'
            X_train, X_val, X_test = s_train, s_val, s_test

            X_train = pd.get_dummies(X_train, columns=X_train.columns.to_list(), dtype='float32')
            X_val = pd.get_dummies(X_val, columns=X_val.columns.to_list(), dtype='float32')
            X_test = pd.get_dummies(X_test, columns=X_test.columns.to_list(), dtype='float32')
            
            scaler = StandardScaler().fit(X_train) 

            X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

            encoder = None

        elif args.encoder=='target':
            assert args.label=='sensitive', 'if args.encoder is target, thus args.label must be sensitive'
            X_train, X_val, X_test = y_train, y_val, y_test

            X_train = pd.get_dummies(X_train, columns=X_train.columns.to_list(), dtype='float32')
            X_val = pd.get_dummies(X_val, columns=X_val.columns.to_list(), dtype='float32')
            X_test = pd.get_dummies(X_test, columns=X_test.columns.to_list(), dtype='float32')
            
            scaler = StandardScaler().fit(X_train) 

            X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

            encoder = None
    
    train_pandas = PandasDataSet_WithEncoder(X=X_train, label=label_train, s=s_train,encoder = encoder, batch_size=batch_size, device=DEVICE)
    val_pandas = PandasDataSet_WithEncoder(X=X_val, label=label_val, s=s_val, encoder = encoder, batch_size=batch_size, device=DEVICE)
    test_pandas = PandasDataSet_WithEncoder(X=X_test, label=label_test, s=s_test, encoder = encoder, batch_size=batch_size, device=DEVICE)

    train_loader = DataLoader(train_pandas, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_pandas, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_pandas, batch_size=batch_size, shuffle=True)

    if args.encoder=='nrl':
        input_size = config_NRL[args.dataset][args.id_setting_encoder]['z_dim']
    elif args.encoder=='laftr':
        input_size = config_LAFTR[args.dataset][args.id_setting_encoder]['z_dim']
    elif args.encoder=='vfae':
        input_size = config_VFAE[args.dataset][args.id_setting_encoder]['z_dim']
    elif args.encoder is None or args.encoder=='constant':
        input_size = data_stats[args.dataset]['n_features']
        # input_size = 10
        args.id_setting_encoder = None
    elif args.encoder=='sensitive':
        input_size = 4 if args.dataset=='compas' else data_stats[args.dataset]['n_protected']
        args.id_setting_encoder = None
    elif args.encoder=='target':
        input_size = data_stats[args.dataset]['n_target']
        args.id_setting_encoder = None

    classifier = MLP(
            input_size = input_size,
            hidden_layers = config_classifiers[args.dataset][args.id_setting_model]['hdims'],
            output_size = output_size,
            activation_hidden=config_classifiers[args.dataset][args.id_setting_model]['activation_hidden'],
            activation_output=config_classifiers[args.dataset][args.id_setting_model]['activation_output'],
            p_dropout=config_classifiers[args.dataset][args.id_setting_model]['p_dropout']
        ).to(DEVICE)

    # Load pretrained classifier if given
    if args.use_pretrained_classifier_target:
        print(f'Use pretrained model set true. Loading pretrained model with {args.id_setting_model} settings...')

        path_base_model = f"results/{args.encoder}/{args.dataset}/{args.scaler}_{args.stratify_by}/classifiers/target_from_{args.encoder}/{args.id_setting_model}" if args.encoder in [None,'constant'] else f"results/{args.encoder}/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_encoder}/classifiers/target_from_{args.encoder}/{args.id_setting_model}"
        checkpoint_val = torch_load(f'{path_base_model}/val_epoch_results.pt')
        loss_val = checkpoint_val['loss_val']
        checkpoint_val = None
        index_model = loss_val.index(min(loss_val))
        loss_val = None
        checkpoint_train_classifier = torch_load(f'{path_base_model}/train_epoch_{index_model}_model.pt')

        # Before load, check if the output layer has the same size than the target (for ex, for compas, the sizes are different)
        classifier.load_state_dict(checkpoint_train_classifier['classifier_state_dict'])

    # Set optimizer
    lr = args.lr_max if args.use_scheduler else args.lr
    
    if args.optimizer=='Adam':
        optimizer = Adam(classifier.parameters(), lr = lr)
    elif args.optimizer=='AdamW':
        optimizer = AdamW(classifier.parameters(), lr = lr)
    elif args.optimizer=='SGD':
        optimizer = SGD(classifier.parameters(), lr = lr)

    # Set scheduler
    if args.use_scheduler:
        n_ep = total_epochs(args.T_0, args.T_mult, args.num_cycles)
        if args.n_epochs!=n_ep:
            print(f'args.epochs ({args.n_epochs}) is not consistent with the required epochs by the scheduler. Updating to {n_ep}.')
            args.n_epochs = n_ep

        scheduler = CosineAnnealingWarmRestarts(
                    optimizer=optimizer,
                    T_0 = args.T_0,
                    T_mult = args.T_mult,
                    eta_min = args.lr_min,
                    last_epoch=-1
                    )
    else:
        scheduler = None

    # Create folders to save files
    if args.encoder in [None, 'constant', 'sensitive', 'target']:
        path_base = f"results/{args.encoder}/{args.dataset}/{args.scaler}_{args.stratify_by}/classifiers/{args.label}_from_{args.encoder}/{args.id_setting_model}/"
        
    else:
        path_base = f"results/{args.encoder}/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_encoder}/classifiers/{args.label}_from_{args.encoder}/{args.id_setting_model}/"
    
    os.makedirs(path_base, exist_ok=True)

    print('Starting train.')
    train_classifier(
        cl=classifier,
        opt=optimizer,
        sch=scheduler,
        train_l=train_loader,
        val_l=val_loader,
        test_l=test_loader,
        path_base=path_base,
        args=args,
        )
    print('Completed')
    print('')