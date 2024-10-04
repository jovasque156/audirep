# Utilites
import os
import argparse
from pandas import concat as pd_concat
from npeet import entropy_estimators as ee

# Load configs
from configs.config_datasets import data_stats
from configs.config_nrl import config_NRL
from configs.config_laftr import config_LAFTR
from configs.config_vfae import config_VFAE

# Import internal libraries
from src.models.encoders.NRL import NRL
from src.models.encoders.VFAE import VFAE
from src.models.encoders.LAFTR import LAFTR
from src.models.base_models.MLP import MLP
from src.models.base_models.VariationalMLP import VariationalMLP
from src.models.data_loader.PandasDataSet import PandasDataSet_WithEncoder

# Torch
from torch.utils.data import DataLoader
from torch import tensor as torch_tensor
from torch import load as torch_load
from torch import cat as torch_cat
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

def estimate_MI(test_l, path_base, args):
    # go through dataset in order to obain the values and save them in just one variable

    all_X = torch_tensor([])
    all_label = torch_tensor([])
    all_s = torch_tensor([])

    for i, (X,label,s) in enumerate(test_l):
        all_X = torch_cat((all_X, X.detach().cpu()), dim=0)
        all_label = torch_cat((all_label, label.view(-1,1).detach().cpu()), dim=0)

    mi = ee.mi(all_X.numpy(), all_label.numpy())
    with open(path_base+f'encoder_{args.id_setting_encoder}.txt', 'w') as f:
        f.write(f'mi:{str(mi)}\n')

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
    
    # Task
    parser.add_argument('--label', type=str, default='target', help='Define the label: target or sensitive')

    # For debugging
    parser.add_argument('--debugging_mode', action='store_true', help='Use for avoid saving progress')
    args = parser.parse_args()

    print(f'Runing: {args}')

    if args.encoder not in ['constant', 'sensitive', 'target']:
        assert not(bool(args.encoder)!=bool(args.id_setting_encoder)), 'Encoder or Id setting of encoder was given, but not both'

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

    batch_size = args.batch_size if args.batch_size else batch_size
    
    print(f'Loading dataset {args.dataset}, with scaler {args.scaler} and stratify by {args.stratify_by}')
    checkpoint = torch_load(f'results/datasets/{args.dataset}_stratifyby_{args.stratify_by}_scaler_{args.scaler}.pt')

    _, _, test_data = checkpoint['train_data'], checkpoint['val_data'], checkpoint['test_data']

    X_test, y_test, s_test = test_data
    
    if args.label=='target':
        label_test = y_test
        output_size = data_stats[args.dataset]['n_target']
    elif args.label=='sensitive':
        label_test = s_test
        output_size = 4 if args.dataset=='compas' else data_stats[args.dataset]['n_protected']

    if args.encoder:
        if args.encoder=='vfae':
            X_test = pd_concat((X_test, s_test), axis=1)
    
    test_pandas = PandasDataSet_WithEncoder(X=X_test, label=label_test, s=s_test, encoder = encoder, batch_size=batch_size, device=DEVICE)

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

    # Create folders to save files
    path_base = f"results/{args.encoder}/{args.dataset}/{args.scaler}_{args.stratify_by}/MI/{args.label}_from_{args.encoder}/"
    
    os.makedirs(path_base, exist_ok=True)

    print('Starting train.')
    estimate_MI(
        test_l=test_loader,
        path_base=path_base,
        args=args,
        )
    print('Completed')
    print('')