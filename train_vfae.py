# Utilities
import argparse
import os
from tqdm import tqdm
# Import dataset
from src.models.data_loader.tabular_datasets import load_and_process_dataset
from src.models.data_loader.PandasDataSet import PandasDataSet
from torch.utils.data import DataLoader

# Import LFR
from src.models.encoders.VFAE import VFAE
from src.losses.VFAEloss import VFAEloss

# Import models
from src.models.base_models.MLP import MLP
from src.models.base_models.VariationalMLP import VariationalMLP

# Import stats and settings
from configs.config_datasets import data_stats
from configs.config_vfae import config_VFAE

# Library from pytorch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import save as torch_save
from torch import load as torch_load
from torch.cuda import is_available as cuda_is_available
from torch import manual_seed

DEVICE = 'cuda' if cuda_is_available() else 'cpu'

SENSITIVES = {
    'compas': 'race',
    'adult': 'sex',
    'census_income_kdd': 'sex'
}

def train_vfae(vfae_model, vfae_optimizer, vfae_scheduler, train_loader, val_loader, args):
    '''
    train vfae given the parameters and return the model. It also saved log files
    '''
    manual_seed(0)

    loss_criteria = VFAEloss(
                        alpha = config_VFAE[args.dataset][args.id_setting_vfae]['alpha'],
                        beta = config_VFAE[args.dataset][args.id_setting_vfae]['beta'],
                        mmd_dim = config_VFAE[args.dataset][args.id_setting_vfae]['mmd_dim'],
                        mmd_gamma = config_VFAE[args.dataset][args.id_setting_vfae]['mmd_gamma'],
                    )

    L_loss_train = []
    L_rec_train = []
    L_kl_z1_train = []
    L_kl_z2_train = []
    L_supervised_train = []
    L_mmd_train = []
    
    L_loss_val = []
    L_rec_val = []
    L_kl_z1_val = []
    L_kl_z2_val = []
    L_supervised_val = []
    L_mmd_val = []

    # ipdb.set_trace()
    iters = 0
    last_best_epoch = 0
    perfo_best = 100000
    steps_per_epoch = len(train_loader)
    with tqdm(range(args.n_epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            vfae_model.train()
            total_loss = 0
            total_rec_loss = 0
            total_kl_loss_z1 = 0
            total_kl_loss_z2 = 0
            total_supervised_loss = 0
            total_loss_mmd = 0
            n_total = 0

            tepoch.set_description(f"Epoch {epoch}")
            
            for i, (X, y, s) in enumerate(train_loader):
                X, y, s = X.to(DEVICE), y.to(DEVICE), s.to(DEVICE), 

                # Train according to the number of fair and dev steps
                n_total += X.shape[0]
                p = int(iters/10) % data_stats[args.dataset]['n_protected'] if data_stats[args.dataset]['n_protected']>2 else 0
                s = 1.0*(s==p)

                if (sum(s==0)/n_total)==1.0 or (sum(s==0)/n_total)==0.0:
                    continue

                vfae_optimizer.zero_grad()

                decoded = vfae_model(X,s,y)

                loss, items = loss_criteria(decoded, X, s, y.view(-1).long())
                loss.backward()

                vfae_optimizer.step()
                if args.use_scheduler:
                    vfae_scheduler.step(epoch + i / steps_per_epoch)

                total_loss += loss.item()
                total_rec_loss += items['rec_loss'].item()*X.shape[0]
                total_kl_loss_z1 += items['kl_loss_z1'].item()*X.shape[0]
                total_kl_loss_z2 += items['kl_loss_z2'].item()*X.shape[0]
                total_supervised_loss += items['supervised_loss'].item()*X.shape[0]
                total_loss_mmd += items['mmd'].item()*X.shape[0]
                n_total += X.shape[0]

                iters+=1
            
            # Log the results so far
            torch_save(
                {
                    'vfae_state_dict': vfae_model.state_dict(),
                    'vfae_structure': str(vfae_model),
                    'vfae_optimizer': vfae_optimizer.state_dict(),
                    'vfae_optimizer_structure': str(vfae_optimizer),
                    'vfae_scheduler': str(vfae_scheduler),
                    'parameters': vars(args)
                },
                f"results/vfae/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_vfae}/train_epoch_{epoch}_model.pt"
            )

            L_loss_train.append(total_loss/n_total)
            L_rec_train.append(total_rec_loss/n_total)
            L_kl_z1_train.append(total_kl_loss_z1/n_total)
            L_kl_z2_train.append(total_kl_loss_z2/n_total)
            L_supervised_train.append(total_supervised_loss/n_total)
            L_mmd_train.append(total_loss_mmd/n_total)

            torch_save(
                {
                    'loss_total': L_loss_train,
                    'loss_rec': L_rec_train,
                    'loss_kl_z1': L_kl_z1_train,
                    'loss_kl_z2': L_kl_z2_train,
                    'loss_supervised': L_supervised_train,
                    'parameters': vars(args)
                },
                f"results/vfae/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_vfae}/train_epoch_results.pt"
            )

            total_loss = 0
            total_rec_loss = 0
            total_kl_loss_z1 = 0
            total_kl_loss_z2 = 0
            total_supervised_loss = 0
            total_loss_mmd = 0
            n_total = 0
            vfae_model.eval()
            for X, y, s in val_loader:
                X, y, s = X.to(DEVICE), y.to(DEVICE), s.to(DEVICE)

                # This is only for computing a random average
                p = int(iters/10) % data_stats[args.dataset]['n_protected'] if data_stats[args.dataset]['n_protected']>2 else 0
                s = 1.0*(s==p)

                decoded = model(X,s,y)

                loss, items = loss_criteria(decoded, X, s, y.view(-1).long())

                total_loss += loss.item()
                total_rec_loss += items['rec_loss'].item()*X.shape[0]
                total_kl_loss_z1 += items['kl_loss_z1'].item()*X.shape[0]
                total_kl_loss_z2 += items['kl_loss_z2'].item()*X.shape[0]
                total_supervised_loss += items['supervised_loss'].item()*X.shape[0]
                total_loss_mmd += items['mmd'].item()*X.shape[0]
                n_total += X.shape[0]

            L_loss_val.append(total_loss/n_total)
            L_rec_val.append(total_rec_loss/n_total)
            L_kl_z1_val.append(total_kl_loss_z1/n_total)
            L_kl_z2_val.append(total_kl_loss_z2/n_total)
            L_supervised_val.append(total_supervised_loss/n_total)
            L_mmd_val.append(total_loss_mmd/n_total)

            tepoch.set_postfix(
                        loss_t=L_loss_train[-1], 
                        loss_rec_t=L_rec_train[-1],
                        loss_kl_z1_t = L_kl_z1_train[-1],
                        loss_kl_z2_t = L_kl_z2_train[-1],
                        loss_supervised_t = L_supervised_train[-1],
                        loss_mmd_t = L_mmd_train[-1],
                        loss_v=L_loss_val[-1], 
                        loss_rec_v= L_rec_val[-1],
                        loss_kl_z1_v = L_kl_z1_val[-1],
                        loss_kl_z2_v = L_kl_z2_val[-1],
                        loss_supervised_v = L_supervised_val[-1],
                        loss_mmd_v = L_mmd_val[-1],
                    )
            print()            
            
            # Log results so far
            if perfo_best - (total_loss/n_total) >= args.stop_epsilon*perfo_best:
                perfo_best = (total_loss/n_total)
                last_best_epoch = epoch

            torch_save(
                {
                    'loss_total': L_loss_val,
                    'loss_rec': L_rec_val,
                    'loss_kl_z1': L_kl_z1_val,
                    'loss_kl_z2': L_kl_z2_val,
                    'loss_supervised': L_supervised_val,
                    'parameters': vars(args),
                    'last_significant_best_epoch': last_best_epoch,
                },
                f"results/vfae/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_vfae}/val_epoch_results.pt"
            )

            if args.early_stop:
                if last_best_epoch<=epoch-args.n_ep_imp:
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset to train vfae')
    parser.add_argument('--stratify_by', type=str, default='target+sensitive', help='Variables to be used for stratification')
    parser.add_argument('--scaler', type=str,default='None', help='Scaler to apply, it can be standard, MinMax, or None')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--optimizer', type=str, help='Optimizer to be used')
    parser.add_argument('--learning_rate_min', type=float, default=0.0, help='Min learning rate for vfae')
    parser.add_argument('--learning_rate_max', type=float, default=5e-5, help='Max learning rate for vfae')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs for vfae')
    parser.add_argument('--create_new_dataset', action='store_true', help='Set to create obligate to create a new dataset, no matter if an older with same settings already exists')
    parser.add_argument('--path_data', type=str, help='Path to data to be used')
    parser.add_argument('--early_stop', action='store_true', help='Stop training if sufficient performance was achieved')
    parser.add_argument('--stop_epsilon', type=float, default=1e-2, help='Set an epsilon if you want to stop training if no improvement is identified after --n_ep_imp (default = 10)')
    parser.add_argument('--n_ep_imp', type=int, default=5, help='Number of epochs with no improvement in the val dataset')
    parser.add_argument('--use_scheduler', action='store_true', help='Use for using scheduler CosineAnnealingWarmRestarts')
    parser.add_argument('--T_0', type=int, default=10, help='Number of epochs for first cycle')
    parser.add_argument('--T_mult', type=int, default=1, help='Multiplier for cycles>1')
    parser.add_argument('--num_cycles', type=int, default=5, help='Number of cycles')

    # Set version of vfae
    parser.add_argument('--id_setting_vfae', type=str, help='Version settings for the VFAE, see config_VFAE')

    args = parser.parse_args()

    manual_seed(0)

    print(f'Runing: {args}')
    
    # Load dataset
    os.makedirs(f'results/datasets/', exist_ok=True)
    if args.path_data:
        print('Data path given. Importing dataset...')
        checkpoint = torch_load(args.path_data)
        train_data = checkpoint['train_data']
        val_data = checkpoint['val_data']
    elif os.path.exists(f'results/datasets/{args.dataset}_stratifyby_{args.stratify_by}_scaler_{args.scaler}.pt') and not args.create_new_dataset:
        print('Data already exists. Importing dataset.')
        checkpoint = torch_load(f'results/datasets/{args.dataset}_stratifyby_{args.stratify_by}_scaler_{args.scaler}.pt')
        train_data = checkpoint['train_data']
        val_data = checkpoint['val_data']
    else:
        print("Data doesn't exist. Creating a new one.")
        train_data, val_data, test_data = load_and_process_dataset(dataset=args.dataset,
                                                    sensitive_attribute=SENSITIVES[args.dataset],
                                                    stratify_by=args.stratify_by,
                                                    scaler=args.scaler)
        torch_save(
            {
                'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data,
            },
            f'results/datasets/{args.dataset}_stratifyby_{args.stratify_by}_scaler_{args.scaler}.pt'
        )

    X_train, y_train, s_train = train_data
    X_val, y_val, s_val = val_data

    train_pandas = PandasDataSet(X_train, y_train, s_train)
    val_pandas = PandasDataSet(X_val, y_val, s_val)

    if not args.batch_size:
        args.batch_size = 254 if args.dataset=='compas' else 512

    train_loader = DataLoader(train_pandas, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_pandas, batch_size=args.batch_size, shuffle=True)

    # Loading model settings
    encoder_z1 = VariationalMLP(in_features= data_stats[args.dataset]['n_features']+data_stats[args.dataset]['s_dim'],
                            hidden_dims= config_VFAE[args.dataset][args.id_setting_vfae]['z1_enc_dim'],
                            z_dim= config_VFAE[args.dataset][args.id_setting_vfae]['z_dim'],
                            activation_hidden = config_VFAE[args.dataset][args.id_setting_vfae]['z1_enc_activation_hidden'],
                            p_dropout = config_VFAE[args.dataset][args.id_setting_vfae]['z1_enc_p_dropout'],
                        )
    encoder_z2 = VariationalMLP(
                                in_features= config_VFAE[args.dataset][args.id_setting_vfae]['z_dim']+data_stats[args.dataset]['y_dim'],
                                hidden_dims= config_VFAE[args.dataset][args.id_setting_vfae]['z2_enc_dim'],
                                z_dim= config_VFAE[args.dataset][args.id_setting_vfae]['z_dim'],
                                activation_hidden = config_VFAE[args.dataset][args.id_setting_vfae]['z2_enc_activation_hidden'],
                                p_dropout = config_VFAE[args.dataset][args.id_setting_vfae]['z2_enc_p_dropout'],
                                )
    decoder_z1 = VariationalMLP(
                                in_features= config_VFAE[args.dataset][args.id_setting_vfae]['z_dim']+data_stats[args.dataset]['y_dim'],
                                hidden_dims= config_VFAE[args.dataset][args.id_setting_vfae]['z1_dec_dim'],
                                z_dim= config_VFAE[args.dataset][args.id_setting_vfae]['z_dim'],
                                activation_hidden = config_VFAE[args.dataset][args.id_setting_vfae]['z1_dec_activation_hidden'],
                                p_dropout = config_VFAE[args.dataset][args.id_setting_vfae]['z1_enc_p_dropout'],
                                )
    y_out_dim = 2 if data_stats[args.dataset]['y_dim']==1 else data_stats[args.dataset]['y_dim']
    decoder_y = MLP(input_size=config_VFAE[args.dataset][args.id_setting_vfae]['z_dim'],
                    hidden_layers=config_VFAE[args.dataset][args.id_setting_vfae]['x_dec_dim'],
                    output_size=y_out_dim,
                    activation_hidden=config_VFAE[args.dataset][args.id_setting_vfae]['x_dec_activation_hidden'],
                    activation_output=config_VFAE[args.dataset][args.id_setting_vfae]['x_dec_activation_output'],
                    p_dropout=config_VFAE[args.dataset][args.id_setting_vfae]['x_dec_p_dropout'])
    #Free memory
    y_out_dim=None

    decoder_x = MLP(input_size=config_VFAE[args.dataset][args.id_setting_vfae]['z_dim']+data_stats[args.dataset]['s_dim'],
                    hidden_layers=config_VFAE[args.dataset][args.id_setting_vfae]['x_dec_dim'],
                    output_size=data_stats[args.dataset]['n_features']+data_stats[args.dataset]['s_dim'],
                    activation_hidden=config_VFAE[args.dataset][args.id_setting_vfae]['x_dec_activation_hidden'],
                    activation_output=config_VFAE[args.dataset][args.id_setting_vfae]['x_dec_activation_output'],
                    p_dropout=config_VFAE[args.dataset][args.id_setting_vfae]['x_dec_p_dropout'])

    model = VFAE(
                encoder_z1=encoder_z1,
                encoder_z2=encoder_z2,
                decoder_z1=decoder_z1,
                decoder_y=decoder_y,
                decoder_x=decoder_x).to(DEVICE)

    steps_per_epoch = len(train_loader)

    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr = args.learning_rate_max)
    elif args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr = args.learning_rate_max)

    if args.use_scheduler:
        scheduler = CosineAnnealingWarmRestarts(
                    optimizer=optimizer,
                    T_0 = args.T_0,
                    T_mult = args.T_mult,
                    eta_min = args.learning_rate_min,
                    last_epoch=-1
                    )
    else:
        scheduler = None

    print(f'Training vfae with {args.id_setting_vfae} settings')
    os.makedirs(f'results/vfae/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_vfae}/', exist_ok=True)
    train_vfae(
        vfae_model = model,
        vfae_optimizer = optimizer,
        vfae_scheduler = scheduler,
        train_loader = train_loader,
        val_loader = val_loader,
        args = args
    )
