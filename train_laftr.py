# Utilities
import argparse
import os
from tqdm import tqdm

# Import dataset
from src.models.data_loader.tabular_datasets import load_and_process_dataset
from src.models.data_loader.PandasDataSet import PandasDataSet
from torch.utils.data import DataLoader

# Import LFR
from src.models.encoders.LAFTR import LAFTR

# Import models
from src.models.base_models.MLP import MLP

# Import stats and settings
from configs.config_datasets import data_stats
from configs.config_laftr import config_LAFTR

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

import ipdb

def train_laftr(laftr_model, fair_optimizer, adv_optimizer, fair_scheduler, adv_scheduler, train_loader, val_loader, args):
    manual_seed(0)

    L_adv_losses_train = []
    L_classif_losses_train = []
    L_rec_losses_train = []
    L_total_losses_train = []

    L_adv_losses_val = []
    L_classif_losses_val = []
    L_rec_losses_val = []
    L_total_losses_val = []

    iters = 0
    last_best_epoch = 0
    perfo_best = 100000
    steps_per_epoch = len(train_loader)
    with tqdm(range(args.n_epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            laftr_model.train()
            total_loss_adv = 0
            total_loss_classif = 0
            total_loss_rec = 0
            n_total = 0

            tepoch.set_description(f"Epoch {epoch}")
            
            for i, (X, y, s) in enumerate(train_loader):
                X, y, s = X.to(DEVICE), y.to(DEVICE), s.to(DEVICE)

                p = int(iters/10) % data_stats[args.dataset]['n_protected'] if data_stats[args.dataset]['n_protected']>2 else 0
                s = 1.0*(s==p)

                # Train according to the number of fair and dev steps
                n_total += X.shape[0]
                for part, steps, optimizer, scheduler, sign in [('fair', args.steps_opt_fair, fair_optimizer, fair_scheduler, 1), ('adv', args.steps_opt_adv, adv_optimizer, adv_scheduler, -1)]:
                    if part=='fair':
                        laftr_model.encoder.train()
                        laftr_model.decoder.train()
                        laftr_model.classifier.train()
                        laftr_model.adversary.eval()
                    else:
                        laftr_model.encoder.eval()
                        laftr_model.decoder.eval()
                        laftr_model.classifier.eval()
                        laftr_model.adversary.train()

                    for _ in range(steps):
                        optimizer.zero_grad()

                        encoded, decoded, classif_logits, adv_logits = laftr_model.forward(X, s, y)

                        loss, L_x, L_y, L_z = laftr_model.loss(X, s, y, encoded, decoded, classif_logits, adv_logits)

                        if sign==1:
                            total_loss_rec += L_x.item()*X.shape[0]/steps
                            total_loss_classif += L_y.item()*X.shape[0]/steps
                        elif sign==-1:
                            total_loss_adv += L_z.item()*X.shape[0]/steps

                        loss = sign*loss

                        loss.backward()
                        optimizer.step()

                    if args.use_scheduler:
                        scheduler.step(epoch + i / steps_per_epoch)
                
                iters+=1

            L_rec_losses_train.append(total_loss_rec/n_total)
            L_classif_losses_train.append(total_loss_classif/n_total)
            L_adv_losses_train.append(total_loss_adv/n_total)
            L_total_losses_train.append(args.coefs_A_x*total_loss_rec/n_total+args.coefs_A_y*total_loss_classif/n_total-args.coefs_A_z*total_loss_adv/n_total)
            
            # Log the results so far
            torch_save(
                {
                    'laftr_state_dict': laftr_model.state_dict(),
                    'laftr_structure': str(laftr_model),
                    'fair_optimizer': fair_optimizer.state_dict(),
                    'fair_optimizer_structure': str(fair_optimizer),
                    'fair_scheduler': str(fair_scheduler),
                    'adv_optimizer': adv_optimizer.state_dict(),
                    'adv_optimizer_structure': str(adv_optimizer),
                    'adv_scheduler': str(adv_scheduler),
                    'parameters': vars(args)
                },
                f"results/laftr/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_laftr}/train_epoch_{epoch}_model.pt"
            )

            torch_save(
                {
                    'loss_adv': L_adv_losses_train,
                    'loss_classif': L_classif_losses_train,
                    'loss_rec': L_rec_losses_train,
                    'loss_total': L_total_losses_train,
                    'parameters': vars(args)
                },
                f"results/laftr/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_laftr}/train_epoch_results.pt"
            )

            total_loss = 0
            total_loss_adv = 0
            total_loss_classif = 0
            total_loss_rec = 0
            n_total = 0
            laftr_model.eval()
            for X, y, s in val_loader:
                X, s, y = X.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
                n_total += X.shape[0]

                p = int(iters/10) % data_stats[args.dataset]['n_protected'] if data_stats[args.dataset]['n_protected']>2 else 0
                s = 1.0*(s==p)

                encoded, decoded, classif_logits, adv_logits = laftr_model.forward(X, s, y)

                loss, L_x_temp, L_y_temp, L_z_temp = laftr_model.loss(X, s, y, decoded, classif_logits, adv_logits)

                total_loss += loss.item()*X.shape[0]
                total_loss_rec += L_x_temp.item()*X.shape[0]
                total_loss_classif += L_y_temp.item()*X.shape[0]
                total_loss_adv += L_z_temp.item()*X.shape[0]

            L_rec_losses_val.append(total_loss_rec/n_total)
            L_classif_losses_val.append(total_loss_classif/n_total)
            L_adv_losses_val.append(total_loss_adv/n_total)
            L_total_losses_val.append(args.coefs_A_x*total_loss_rec/n_total+args.coefs_A_y*total_loss_classif/n_total-args.coefs_A_z*total_loss_adv/n_total)

            tepoch.set_postfix(
                        loss_adv_t = L_adv_losses_train[-1], 
                        loss_classif_t = L_classif_losses_train[-1],
                        loss_rec_t = L_rec_losses_train[-1],
                        loss_adv_v=L_adv_losses_val[-1], 
                        loss_classif_v=L_classif_losses_val[-1],
                        loss_rec_v = L_rec_losses_val[-1]
                    )   
            
            # Log results so far
            if perfo_best - (total_loss/n_total) >= args.stop_epsilon*perfo_best:
                perfo_best = (total_loss/n_total)
                last_best_epoch = epoch

            torch_save(
                {
                    'loss_adv': L_adv_losses_val,
                    'loss_classif': L_classif_losses_val,
                    'loss_rec': L_rec_losses_val,
                    'loss_total': L_total_losses_val,
                    'parameters': vars(args),
                    'last_significant_best_epoch': last_best_epoch,
                },
                f"results/laftr/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_laftr}/val_epoch_results.pt"
            )

            if args.early_stop:
                if last_best_epoch<=epoch-args.n_ep_imp:
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset to train laftr')
    parser.add_argument('--stratify_by', type=str, default='target+sensitive', help='Variables to be used for stratification')
    parser.add_argument('--scaler', type=str,default='None', help='Scaler to apply, it can be standard, MinMax, or None')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--optimizer', type=str, help='Optimizer to be used')
    parser.add_argument('--learning_rate_fair_min', type=float, default=5e-5, help='Min learning rate for fair (enc-class-dec) architecture from laftr')
    parser.add_argument('--learning_rate_fair_max', type=float, default=0.0, help='Max learning rate for fair (enc-class-dec) architecture from laftr')
    parser.add_argument('--learning_rate_adv_min', type=float, default=5e-5, help='Min learning rate for fair adv architecture from laftr')
    parser.add_argument('--learning_rate_adv_max', type=float, default=0.0, help='Max learning rate for fair adv architecture from laftr')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs for laftr')
    parser.add_argument('--steps_opt_fair', type=int, default=1,help='Steps for enc-dec-classifier')
    parser.add_argument('--steps_opt_adv', type=int, default=1, help='Steps for adversarial')
    parser.add_argument('--coefs_A_x', type=float, default=1.0, help='Value for A_x coef')
    parser.add_argument('--coefs_A_y', type=float, default=1.0, help='Value for A_y coef')
    parser.add_argument('--coefs_A_z', type=float, default=50.0, help='Value for A_z coef')
    parser.add_argument('--create_new_dataset', action='store_true', help='Set to create obligate to create a new dataset, no matter if an older with same settings already exists')
    parser.add_argument('--path_data', type=str, help='Path to data to be used')
    parser.add_argument('--early_stop', action='store_true', help='Stop training if sufficient performance was achieved')
    parser.add_argument('--stop_epsilon', type=float, default=1e-2, help='Set an epsilon if you want to stop training if no improvement is identified after --n_ep_imp (default = 10)')
    parser.add_argument('--n_ep_imp', type=int, default=5, help='Number of epochs with no improvement in the val dataset')
    parser.add_argument('--use_scheduler', action='store_true', help='Use to set on the scheduler')
    parser.add_argument('--T_0', type=int, default=10, help='Number of epochs for first cycle')
    parser.add_argument('--T_mult', type=int, default=1, help='Multiplier for cycles>1')
    parser.add_argument('--num_cycles', type=int, default=5, help='Number of cycles')

    # Set version of laftr
    parser.add_argument('--id_setting_laftr', type=str, help='Version settings for the LAFTR, see config_LAFTR')

    args = parser.parse_args()

    manual_seed(0)
    
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
    encoder = MLP(
                input_size = data_stats[args.dataset]['n_features'],
                hidden_layers = config_LAFTR[args.dataset][args.id_setting_laftr]['encoder_hdims'], 
                output_size = config_LAFTR[args.dataset][args.id_setting_laftr]['z_dim'], 
                activation_hidden = config_LAFTR[args.dataset][args.id_setting_laftr]['encoder_activation_hidden'], 
                activation_output = config_LAFTR[args.dataset][args.id_setting_laftr]['encoder_activation_output'], 
                p_dropout=config_LAFTR[args.dataset][args.id_setting_laftr]['encoder_p_dropout'])
                
    decoder = MLP(
                input_size = config_LAFTR[args.dataset][args.id_setting_laftr]['z_dim'],
                hidden_layers = config_LAFTR[args.dataset][args.id_setting_laftr]['decoder_hdims'], 
                output_size = data_stats[args.dataset]['n_features'], 
                activation_hidden = config_LAFTR[args.dataset][args.id_setting_laftr]['decoder_activation_hidden'], 
                activation_output = config_LAFTR[args.dataset][args.id_setting_laftr]['decoder_activation_output'],
                p_dropout=config_LAFTR[args.dataset][args.id_setting_laftr]['decoder_p_dropout'])
                        
    adversary = MLP(
                input_size = config_LAFTR[args.dataset][args.id_setting_laftr]['z_dim'], 
                hidden_layers = config_LAFTR[args.dataset][args.id_setting_laftr]['adversary_hdims'], 
                output_size =  data_stats[args.dataset]['n_protected'], 
                activation_hidden = config_LAFTR[args.dataset][args.id_setting_laftr]['adversary_activation_hidden'], 
                activation_output = config_LAFTR[args.dataset][args.id_setting_laftr]['adversary_activation_output'], 
                p_dropout=config_LAFTR[args.dataset][args.id_setting_laftr]['adversary_p_dropout'])
                
    classifier = MLP(
                input_size = config_LAFTR[args.dataset][args.id_setting_laftr]['z_dim'], 
                hidden_layers = config_LAFTR[args.dataset][args.id_setting_laftr]['classifier_hdims'], 
                output_size = data_stats[args.dataset]['n_target'], 
                activation_hidden = config_LAFTR[args.dataset][args.id_setting_laftr]['classifier_activation_hidden'], 
                activation_output = config_LAFTR[args.dataset][args.id_setting_laftr]['classifier_activation_output'], 
                p_dropout=config_LAFTR[args.dataset][args.id_setting_laftr]['classifier_p_dropout'])

    laftr_model = LAFTR(
                    encoder = encoder,
                    decoder = decoder,
                    adversary = adversary,
                    classifier = classifier,
                    A_x=args.coefs_A_x,
                    A_y=args.coefs_A_y,
                    A_z=args.coefs_A_z,
                    device=DEVICE
                ).to(DEVICE)

    steps_per_epoch = len(train_loader)

    if args.optimizer == 'SGD':
        fair_optimizer = SGD(laftr_model.fair_parameters(), lr = args.learning_rate_fair_max)
        adv_optimizer = SGD(laftr_model.adv_parameters(), lr = args.learning_rate_adv_max)
    elif args.optimizer == 'Adam':
        fair_optimizer = Adam(laftr_model.fair_parameters(), lr = args.learning_rate_fair_max)
        adv_optimizer = Adam(laftr_model.adv_parameters(), lr = args.learning_rate_adv_max)

    fair_scheduler = None
    adv_scheduler = None
    if args.use_scheduler:
        fair_scheduler = CosineAnnealingWarmRestarts(
                                                        optimizer=fair_optimizer,
                                                        T_0=args.T_0,
                                                        T_mult=args.T_mult,
                                                        eta_min = args.learning_rate_fair_min,
                                                        last_epoch=-1
                                                    )

        adv_scheduler = CosineAnnealingWarmRestarts(
                                                        optimizer=adv_optimizer,
                                                        T_0=args.T_0,
                                                        T_mult=args.T_mult,
                                                        eta_min = args.learning_rate_adv_min,
                                                        last_epoch=-1
                                                    )

    print(f'Training laftr with {args.id_setting_laftr} settings')
    os.makedirs(f'results/laftr/{args.dataset}/{args.scaler}_{args.stratify_by}/{args.id_setting_laftr}/', exist_ok=True)
    train_laftr(
        laftr_model = laftr_model,
        fair_optimizer = fair_optimizer,
        adv_optimizer = adv_optimizer,
        fair_scheduler = fair_scheduler,
        adv_scheduler = adv_scheduler,
        train_loader = train_loader,
        val_loader = val_loader,
        args = args
    )
