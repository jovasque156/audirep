# Utilities
import argparse
import os
from tqdm import tqdm
from numpy import abs as np_abs

# Import dataset
from src.models.data_loader.tabular_datasets import load_and_process_dataset
from src.models.data_loader.PandasDataSet import PandasDataSet
from torch.utils.data import DataLoader

# Import LFR
from src.models.encoders.NRL import NRL

# Import models
from src.models.base_models.MLP import MLP

# Import stats and settings
from configs.config_datasets import data_stats
from configs.config_nrl import config_NRL

# Library from pytorch
from torch.nn import Linear
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import sum as torch_sum 
from torch import pow as torch_pow
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

# from: 
def train_nrl(nrl_model, enc_optimizer, dec_optimizer, crit_optimizer, enc_scheduler, dec_scheduler, crit_scheduler, train_loader, val_loader, args):
    manual_seed(0)

    L_decoder_losses_train = []
    L_critic_losses_train = []
    L_total_losses_train = []

    L_decoder_losses_val = []
    L_critic_losses_val = []
    L_total_losses_val = []

    iters = 0
    last_best_epoch = 0
    perfo_best = 100000
    steps_per_epoch = len(train_loader)
    with tqdm(range(args.n_epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            nrl_model.train()
            dec_losses = 0
            critic_losses = 0
            total_dec = 0
            total_critic = 0
            tepoch.set_description(f"Epoch {epoch}")
            
            for i, (X, y, s) in enumerate(train_loader):
                X, y, s = X.to(DEVICE), y.to(DEVICE), s.to(DEVICE), 

                # Identify the iteration number if # groups >2
                p = int(iters/args.change_group_freq) % data_stats[args.dataset]['n_protected'] if data_stats[args.dataset]['n_protected']>2 else 0
                
                # Control if one of the group is empty
                if sum(s.view(-1)==p) < 2 or sum(s.view(-1)!=p) < 2:
                    continue

                # Train the critic until it converges or a given number of iteratoins
                if args.until_converges:
                    w_dist_last = 0
                    eps = 1
                    
                    while eps >= 1e-3:
                        crit_optimizer[p].zero_grad()

                        w_dist, _ = nrl_model.wdist(nrl_model.encoder(X), s, p)

                        loss = -w_dist
                        loss.backward(retain_graph=True)

                        crit_optimizer[p].step()

                        for params in nrl_model.critic[p].parameters():
                            params.data.clamp_(-0.1, 0.1)

                        # keep training crit until distance no longer decrease
                        eps = np_abs(w_dist.data.item() - w_dist_last)
                        w_dist_last = w_dist.data.item()
                else:
                    for _ in range(args.c_iter):
                        crit_optimizer[p].zero_grad()
                        
                        w_dist, _ = nrl_model.wdist(nrl_model.encoder(X), s, p)
                        
                        loss = -w_dist
                        loss.backward(retain_graph=True)

                        crit_optimizer[p].step()

                        for params in nrl_model.critic[p].parameters():
                            params.data.clamp_(-0.1, 0.1)
                
                if args.use_scheduler:
                    for crit_sched in crit_scheduler:
                        crit_sched.step(epoch + i / steps_per_epoch)

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                mse_mean, mse_counts, w_dist_mean, w_dist_counts = nrl_model.forward(X, s, p)

                loss = mse_mean + args.alpha*w_dist_mean

                # Saving results
                total_dec += mse_counts
                dec_losses += mse_mean.item()*mse_counts
                critic_losses += w_dist_mean.item()*w_dist_counts
                total_critic += w_dist_counts

                # Adding regularization term
                reg_loss = 0
                for layer in nrl_model.encoder.network:
                    if type(layer) is Linear:
                        norm = 0.0
                        for row in layer.weight.transpose(0,1):
                            norm += torch_sum(torch_pow(row, 2))
                        reg_loss += norm


                # Adding regularization term to the total loss
                loss += args.C_reg * reg_loss

                loss.backward(retain_graph=True)

                enc_optimizer.step()
                dec_optimizer.step()

                if args.use_scheduler:
                    enc_scheduler.step(epoch + i / steps_per_epoch)
                    dec_scheduler.step(epoch + i / steps_per_epoch)

                iters+=1

            L_decoder_losses_train.append(dec_losses/total_dec)
            L_critic_losses_train.append(critic_losses/total_critic)
            L_total_losses_train.append(dec_losses/total_dec+args.alpha*critic_losses/total_critic)

            # Log the results so far
            torch_save(
                {
                    'nrl_state_dict': nrl_model.state_dict(),
                    'nrl_structure': str(nrl_model),
                    'enc_optimizer': enc_optimizer.state_dict(),
                    'enc_optimizer_structure': str(enc_optimizer),
                    'enc_scheduler': str(enc_scheduler) if args.use_scheduler else '',
                    'dec_optimizer': dec_optimizer.state_dict(),
                    'dec_optimizer_structure': str(dec_optimizer) if args.use_scheduler else '',
                    'dec_scheduler': str(dec_scheduler) if args.use_scheduler else '',
                    'crit_optimizer': [crit.state_dict() for crit in crit_optimizer],
                    'crit_optimizer_structure': [str(crit) for crit in crit_optimizer],
                    'crit_scheduler': [str(crit) for crit in crit_scheduler] if args.use_scheduler else '',
                    'parameters': vars(args)
                },
                f"results/nrl/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_nrl}/train_epoch_{epoch}_model.pt"
            )

            torch_save(
                {
                    'loss_dec': L_decoder_losses_train,
                    'loss_crit': L_critic_losses_train,
                    'loss_total': L_total_losses_train,
                    'parameters': vars(args)
                },
                f"results/nrl/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_nrl}/train_epoch_results.pt"
            )

            mse_total = 0
            wdist_total = 0
            total_mse = 0
            total_wdist = 0
            nrl_model.eval()
            p = 0
            for X, y, s in val_loader:
                X, s, y = X.to(DEVICE), s.to(DEVICE), y.to(DEVICE)

                if sum(s.view(-1)==p) < 2 or sum(s.view(-1)!=p) <2:
                    continue

                mse_mean, mse_counts, _, _ = nrl_model.forward(X, s, p)

                mse_total += mse_mean.item()*mse_counts
                total_mse += mse_counts

                all_p = 1 if data_stats[args.dataset]['n_protected']<=2 else data_stats[args.dataset]['n_protected']
                for p in range(all_p):
                    if sum(s.view(-1)==p) < 2 or sum(s.view(-1)!=p) <2:
                        continue
                                
                    w_dist_mean, w_dist_counts = nrl_model.wdist(nrl_model.encoder(X), s, p)

                    wdist_total += w_dist_mean.item()*w_dist_counts
                    total_wdist += w_dist_counts

            L_decoder_losses_val.append(mse_total/total_mse)
            L_critic_losses_val.append(wdist_total/total_wdist)
            L_total_losses_val.append(mse_total/total_mse+args.alpha*wdist_total/total_wdist)

            tepoch.set_postfix(
                        loss_dec_t=L_decoder_losses_train[-1], 
                        loss_crit_t=L_critic_losses_train[-1],
                        loss_dec_v=L_decoder_losses_val[-1],
                        loss_crit_v=L_critic_losses_val[-1]
                    )
            
            # Log results so far
            if perfo_best - (mse_total/total_mse + args.alpha*wdist_total/total_wdist) >= args.stop_epsilon*perfo_best:
                perfo_best = mse_total/total_mse + args.alpha*wdist_total/total_wdist
                last_best_epoch = epoch

            torch_save(
                {
                    'loss_dec': L_decoder_losses_val,
                    'loss_crit': L_critic_losses_val,
                    'loss_total': L_total_losses_val,
                    'parameters': vars(args),
                    'last_significant_best_epoch': last_best_epoch,
                },
                f"results/nrl/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_nrl}/val_epoch_results.pt"
            )

            if args.early_stop:
                if last_best_epoch<=epoch-args.n_ep_imp:
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset to train nrl')
    parser.add_argument('--stratify_by', type=str, default='target+sensitive', help='Variables to be used for stratification')
    parser.add_argument('--scaler', type=str,default='None', help='Scaler to apply, it can be standard, MinMax, or None')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--optimizer', type=str, help='Optimizer to be used')
    parser.add_argument('--learning_rate_min', type=float, default=1e-6, help='Min learning rate for nrl')
    parser.add_argument('--learning_rate_max', type=float, default=0.01, help='Min learning rate for nrl')
    parser.add_argument('--learning_rate_critic_min', type=float, default=1e-6, help='Max learning rate for nrl')
    parser.add_argument('--learning_rate_critic_max', type=float, default=0.01, help='Max learning rate for nrl')
    parser.add_argument('--n_epochs', type=int, default=350, help='Number of epochs for nrl')
    parser.add_argument('--until_converges', action='store_true', help='Set true the train critics until convergences with epsilon=1e-3')
    parser.add_argument('--change_group_freq', type=int, default=10, help='After how many iterations the p is change to the next group for critic')
    parser.add_argument('--c_iter', type=int, default=10, help='Number of iterations of critic')
    parser.add_argument('--alpha', type=float, default=10.0, help='Coefficient alpha')
    parser.add_argument('--C_reg', type=float, default=1.0, help='Regularization term')
    parser.add_argument('--create_new_dataset', action='store_true', help='Set to create obligate to create a new dataset, no matter if an older with same settings already exists')
    parser.add_argument('--path_data', type=str, help='Path to data to be used')
    parser.add_argument('--early_stop', action='store_true', help='Stop training if sufficient performance was achieved')
    parser.add_argument('--stop_epsilon', type=float, default=1e-2, help='Set an epsilon if you want to stop training if no improvement is identified after --n_ep_imp (default = 10)')
    parser.add_argument('--n_ep_imp', type=int, default=5, help='Number of epochs with no improvement in the val dataset')
    parser.add_argument('--use_scheduler', action='store_true', help='Use to set on the scheduler')
    parser.add_argument('--T_0', type=int, default=10, help='Number of epochs for first cycle')
    parser.add_argument('--T_mult', type=int, default=1, help='Multiplier for cycles>1')
    parser.add_argument('--num_cycles', type=int, default=5, help='Number of cycles')

    # Set version of NRL
    parser.add_argument('--id_setting_nrl', type=str, help='Version settings for the NRL, see config_NRL')

    args = parser.parse_args()

    manual_seed(0)
    if args.until_converges:
        print('Deactivating args.c_iter since until_converges set true')
        args.c_iter = None
    
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

    train_loader = DataLoader(train_pandas, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_pandas, batch_size=args.batch_size, shuffle=True)

    # Loading model settings
    encoder = MLP(
                input_size = data_stats[args.dataset]['n_features'], 
                hidden_layers=config_NRL[args.dataset][args.id_setting_nrl]['encoder_hdims'],
                output_size=config_NRL[args.dataset][args.id_setting_nrl]['z_dim'],
                activation_hidden=config_NRL[args.dataset][args.id_setting_nrl]['encoder_activation_hidden'],
                activation_output=config_NRL[args.dataset][args.id_setting_nrl]['encoder_activation_output']
            )
    decoder = MLP(
                input_size = config_NRL[args.dataset][args.id_setting_nrl]['z_dim'],
                hidden_layers = config_NRL[args.dataset][args.id_setting_nrl]['decoder_hdims'],
                output_size=data_stats[args.dataset]['n_features'],
                activation_hidden=config_NRL[args.dataset][args.id_setting_nrl]['decoder_activation_hidden'],
                activation_output=config_NRL[args.dataset][args.id_setting_nrl]['decoder_activation_output'] 
            )

    if data_stats[args.dataset]['n_protected']>2:
        critic = [
                MLP(
                    input_size = config_NRL[args.dataset][args.id_setting_nrl]['z_dim'],
                    hidden_layers = config_NRL[args.dataset][args.id_setting_nrl]['critic_hdims'],
                    output_size=1,
                    activation_hidden=config_NRL[args.dataset][args.id_setting_nrl]['critic_activation_hidden'],
                    activation_output=config_NRL[args.dataset][args.id_setting_nrl]['critic_activation_output'] 
                )
            ]*data_stats[args.dataset]['n_protected']
    else:
        critic = [
                MLP(
                    input_size = config_NRL[args.dataset][args.id_setting_nrl]['z_dim'],
                    hidden_layers = config_NRL[args.dataset][args.id_setting_nrl]['critic_hdims'],
                    output_size=1,
                    activation_hidden=config_NRL[args.dataset][args.id_setting_nrl]['critic_activation_hidden'],
                    activation_output=config_NRL[args.dataset][args.id_setting_nrl]['critic_activation_output'] 
                )
            ]

    nrl_model = NRL(
                    encoder = encoder,
                    decoder = decoder,
                    critic = critic,
                    device=DEVICE
                ).to(DEVICE)

    steps_per_epoch = len(train_loader)

    if args.optimizer == 'SGD':
        enc_optimizer = SGD(nrl_model.encoder.parameters(), lr = args.learning_rate_max)
        dec_optimizer = SGD(nrl_model.decoder.parameters(), lr = args.learning_rate_max)
        crit_optimizer = []
        for critic_nn in nrl_model.critic:
            crit_optimizer.append(SGD(critic_nn.parameters(), lr = args.learning_rate_critic_max))
    elif args.optimizer == 'Adam':
        enc_optimizer = Adam(nrl_model.encoder.parameters(), lr = args.learning_rate_max)
        dec_optimizer = Adam(nrl_model.decoder.parameters(), lr = args.learning_rate_max)
        crit_optimizer = []
        for critic_nn in nrl_model.critic:
            crit_optimizer.append(Adam(critic_nn.parameters(), lr = args.learning_rate_critic_max))

    if args.use_scheduler:
        enc_scheduler = CosineAnnealingWarmRestarts(
                                                        optimizer=enc_optimizer,
                                                        T_0=args.T_0,
                                                        T_mult=args.T_mult,
                                                        eta_min = args.learning_rate_min,
                                                        last_epoch=-1
                                                    )

        dec_scheduler = CosineAnnealingWarmRestarts(
                                                        optimizer=dec_optimizer,
                                                        T_0=args.T_0,
                                                        T_mult=args.T_mult,
                                                        eta_min = args.learning_rate_min,
                                                        last_epoch=-1
                                                    )

        crit_scheduler = []
        for critc in crit_optimizer:
            crit_scheduler.append(CosineAnnealingWarmRestarts(
                                                        optimizer=critc,
                                                        T_0=args.T_0,
                                                        T_mult=args.T_mult,
                                                        eta_min = args.learning_rate_critic_min,
                                                        last_epoch=-1
                                                    )
                                )
    else:
        enc_scheduler = dec_scheduler = crit_scheduler = None

    print(f'Training nrl with {args.id_setting_nrl} settings')
    os.makedirs(f'results/nrl/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_nrl}/', exist_ok=True)
    train_nrl(
        nrl_model = nrl_model,
        enc_optimizer = enc_optimizer,
        dec_optimizer = dec_optimizer,
        crit_optimizer = crit_optimizer,
        enc_scheduler = enc_scheduler,
        dec_scheduler = dec_scheduler,
        crit_scheduler = crit_scheduler,
        train_loader = train_loader,
        val_loader = val_loader,
        args = args
    )
