import os
import yaml
import wandb
import argparse
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data_loader1 import PCADataModule
from model1 import PCAModel
# from codes.data_loader1 import PCADataModule
# from codes.model1 import PCAModel
# from train.model import PCAModel
# from train.data_loader import PCADataModule
torch.set_default_dtype(torch.float64)


if __name__ == '__main__':
    # IOdata file 

    # IOdata_fnm0 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_Idata=coeffs_nl=1e-03_Dsize=1M_pH=1_absB=1_hcosTh=1.pickle'
    IOdata_fnm0 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_nC=9532_nl=1e-03_pH=1_Btan=1_hsinTh=1.pickle'
    # IOdata_fnm0 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_nl=1e-03_Dsize=1M_pH=1_Btan=1_hcosTh=1.pickle'
    # IOdata_fnm0 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_nl=1e-03_Dsize=1M_pH=1_Btan=1_hsinTh=1.pickle'

    # Blin models
    # IOdata_fnm0 = 'Jul26_pcaDB_LHS_2M_Blin_1st10k_MgIIhk_Obs_nPSF_nC=9532_nl=1e-03_Dsize=1M_pH=1_Btan=1_hsinTh=1.pickle'
    # IOdata_fnm0 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_nC=9532_nl=1e-03_Dsize=1M_pH=1_Btan=1_habscosTh=1.pickle'
    # IOdata_fnm0 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_nC=9532_nl=1e-03_Dsize=4M_pH=1_Btan=1_habscosTh=1.pickle'
    # IOdata_fnm0 = 'Jul26_pcaDB_LHS_2M_Blin_1st10k_MgIIhk_Obs_nPSF_nl=1e-03_Dsize=1M_pH=1_Btan=1_hsinTh=1.pickle'
    # Monte carlo with profile filtering if chisqr too small
    # IOdata_fnm0 = 'MgIIhk_PCAdbase_nPSF_may17_1st10k_MgIIhk_Obs_nPSF_nC=9532_nl=1e-03_Dsize=1M_pH=1_Btan=1_hsinTh=1.pickle'
    print(f'\n IOdata_fnm0={IOdata_fnm0} \n')
    # -----------------------------------------
    # common folders

    # parent_dir = '/glade/u/home/piyushag/cmex_ml0'
    # IOdata_fold = f'{parent_dir}/NN_IOdata_ckpts_Jul11onwards/NN_IOdata'  
    parent_dir = '/glade/work/piyushag/cmex_ml0'
    NN_IOdir0  = 'NN_IOdata_ckpts_Jul26onwards'
    # ------------------------------------------
    IOdata_fold = f'{parent_dir}/{NN_IOdir0}/NN_IOdata'
    config_fnm = f'{parent_dir}/codes/config.yaml'

    IOdata_fnm = f"{IOdata_fold}/{IOdata_fnm0}"
    IOdata_fnm00 = IOdata_fnm0.split('.pickle')[0]
    
    for tDBsize_in_M in [1., ]:

        # ---------------------
        with open(config_fnm, 'r') as stream:
            config_data = yaml.load(stream, Loader=yaml.SafeLoader)
        maxEpoch = config_data['max_epochs']

        ckpt_fold = f'{parent_dir}/{NN_IOdir0}/ckpts_fold/{IOdata_fnm00}_mEpch={maxEpoch}_DBsize={tDBsize_in_M:.1f}M' # to store checkpoints during training
        void = os.makedirs(ckpt_fold)
        
        # ---------------------
        # Seed: For reproducibility
        seed = config_data['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ------------------------------------------
        # Initialize data loader
        data_loader = PCADataModule(data_fnm=IOdata_fnm, 
                                    seed=seed, num_workers=os.cpu_count() // 2,
                                    key_add_noise_to_Idata=0, noise_level=0, 
                                    DBsize_in_M=tDBsize_in_M)
        data_loader.setup() # Generate training/validation/test sets
        
        # ------------------------------------------
        # stats for input and output parameters - mean, std
        # io_stats = data_loader.io_stats

        # Initialize model  # TODO: Modify
        model = PCAModel(
                         # n_input=config_data['n_input'], # number of inputs
                         # n_output=config_data['n_output'], # number of outputs
                         n_input  = data_loader.io_stats['mean_Idata'].size, # input data nvars
                         n_output = data_loader.io_stats['mean_Odata'].size, # output data nvars
                         
                         io_stats = data_loader.io_stats, # mean/std of IO data
                         key_add_noise_to_Idata=0,
                         noise_level=0,

                         n_hidden_layers  = config_data['n_hidden_layers'],
                         n_hidden_neurons = config_data['n_hidden_neurons'],
                         lr = config_data['lr'])

        # Initialize logger
        wandb_logger = WandbLogger(entity=config_data['wandb']['entity'],
                                   project=config_data['wandb']['project'],
                                   group=config_data['wandb']['group'],
                                   job_type=config_data['wandb']['job_type'],
                                   tags=config_data['wandb']['tags'],
                                   name=IOdata_fnm00,
                                   notes=config_data['wandb']['notes'],
                                   config=config_data)

        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_fold, # checkpoint_dir,
                                  monitor='valid_loss', mode='min', 
                                  filename=f'best_Epch={epoch:03d}', # {valid_loss:.2f}
                                  save_top_k=1,   # saves the top model that leads to best validation loss
                                  save_last=True, # saves the current state of the training, to help Resume from last saved ckpt file. 
                                 )

        # Initialize trainer
        trainer = Trainer(default_root_dir=ckpt_fold, # checkpoint_dir,
                          accelerator="gpu",
                          devices=1 if torch.cuda.is_available() else None,
                          max_epochs=maxEpoch,
                          callbacks=[checkpoint_callback],
                          logger=wandb_logger,
                          log_every_n_steps=10, 
                          )

        # Train the model ⚡
        trainer.fit(model, data_loader, ckpt_path='last')

        # Save model and 
        save_dictionary = config_data
        save_dictionary['model'] = model

        finalcheckpoint_path = f"{os.path.dirname(IOdata_fold)}/{os.path.basename(ckpt_fold)}.ckpt"
        # finalcheckpoint_path = f"{os.path.dirname(IOdata_fold)}/{checkpoint_fnm0}_mEpch={maxEpoch}_DBsz={tDBsize_in_M}M.ckpt"
        # finalcheckpoint_path = f"{os.path.dirname(IOdata_fold)}/{checkpoint_fnm0}_mE={maxEpoch}.ckpt"
        torch.save(save_dictionary, finalcheckpoint_path)

        # Evaluate on test set
        # Load model from checkpoint
        state = torch.load(finalcheckpoint_path)
        model = state['model']
        trainer.test(model, data_loader)

        # Finalize logging
        wandb.finish()
