import os
import yaml
import wandb
import argparse
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from codes.data_loader1 import PCADataModule
from codes.model1 import PCAModel
# from train.model import PCAModel
# from train.data_loader import PCADataModule
torch.set_default_dtype(torch.float64)


if __name__ == '__main__':
    # input
    # IOdata_fnm0 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_Idata=coeffs_nl=1e-03_Dsize=1M_pH=1_absB=1_hcosTh=1.pickle'
    # IOdata_fnm0 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_nl=1e-03_Dsize=1M_pH=1_Btan=1_hcosTh=1.pickle'
    # IOdata_fnm0 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_nl=1e-03_Dsize=1M_pH=1_Btan=1_hsinTh=1.pickle'
    # blin models
    # IOdata_fnm0 = 'Jul26_pcaDB_LHS_2M_Blin_1st10k_MgIIhk_Obs_nPSF_nC=9532_nl=1e-03_Dsize=1M_pH=1_Btan=1_hsinTh=1.pickle'
    # IOdata_fnm0 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_nC=9532_nl=1e-03_Dsize=1M_pH=1_Btan=1_habscosTh=1.pickle'
    IOdata_fnm0 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_nC=9532_nl=1e-03_Dsize=4M_pH=1_Btan=1_habscosTh=1.pickle'
    # IOdata_fnm0 = 'Jul26_pcaDB_LHS_2M_Blin_1st10k_MgIIhk_Obs_nPSF_nl=1e-03_Dsize=1M_pH=1_Btan=1_hsinTh=1.pickle'
    # Monte carlo with profile filtering if chisqr too small
    # IOdata_fnm0 = 'MgIIhk_PCAdbase_nPSF_may17_1st10k_MgIIhk_Obs_nPSF_nC=9532_nl=1e-03_Dsize=1M_pH=1_Btan=1_hsinTh=1.pickle'

    print(f'\n IOdata_fnm0={IOdata_fnm0} \n')
    # ------------------------------------------
    
    key_add_noise_to_Idata =0 # Add noise to IData during training/validation
    noise_level = 1e-3 # noise level -as per work: "Jul3_howmuch_indiv_coeffs_change_for_a_given_noiselevel.ipynb"

    if not 'nl=0' in IOdata_fnm0:
        if key_add_noise_to_Idata ==1:
            raise NameError('Shouldnt have noise in coeffs and also add additional noise during traning. Fix this.')

    
    # ------------------------------------------
    # parent_dir = '/glade/u/home/piyushag/cmex_ml0'
    # IOdata_fold = f'{parent_dir}/NN_IOdata_ckpts_Jul11onwards/NN_IOdata'    
    parent_dir = '/glade/work/piyushag/cmex_ml0'
    IOdata_fold = f'{parent_dir}/NN_IOdata_ckpts_Jul26onwards/NN_IOdata'      
    config_fnm = f"{parent_dir}/config.yaml"
    
    
    # ---------------------
    IOdata_fnm = f"{IOdata_fold}/{IOdata_fnm0}"
    
    checkpoint_fnm0 = IOdata_fnm0.split('.pickle')[0]
    if key_add_noise_to_Idata ==1:
        checkpoint_fnm0 = f'{checkpoint_fnm0}_nlC={noise_level:.0e}'
    else:
        checkpoint_fnm0 = f'{checkpoint_fnm0}_nlC=0'
        noise_level=0.
    
    checkpoint_dir = f'{os.path.dirname(IOdata_fold)}/tckpts'
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(config_fnm, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)
    
    # Seed: For reproducibility
    seed = config_data['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    maxEpoch = config_data['max_epochs']
    checkpoint_fnm0 = f'{checkpoint_fnm0}_maxEpoch={maxEpoch}'

    # nmodel_range=(10000, 250000) # start, end index of model to train over
    print(checkpoint_fnm0.split('nPSF_'))
    tstr1 = checkpoint_fnm0.split('nPSF_')[1]
    wandb_name = f'PCA_|costheta| - {tstr1}'
    wandb_name = IOdata_fnm0
    # ------------------------------------------
    # Initialize data loader
    data_loader = PCADataModule(data_fnm=IOdata_fnm, 
                                seed=seed, num_workers=os.cpu_count() // 2,
                                key_add_noise_to_Idata=0,
                                noise_level=noise_level)
    # Generate training/validation/test sets
    data_loader.setup()
    
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
                     key_add_noise_to_Idata=key_add_noise_to_Idata,
                     noise_level=noise_level,

                     n_hidden_layers  = config_data['n_hidden_layers'],
                     n_hidden_neurons = config_data['n_hidden_neurons'],
                     lr = config_data['lr'])

    # Initialize logger
    wandb_logger = WandbLogger(entity=config_data['wandb']['entity'],
                               project=config_data['wandb']['project'],
                               group=config_data['wandb']['group'],
                               job_type=config_data['wandb']['job_type'],
                               tags=config_data['wandb']['tags'],
                               name=wandb_name,
                               notes=config_data['wandb']['notes'],
                               config=config_data)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                          monitor='valid_loss', mode='min', save_top_k=1,
                                          filename=checkpoint_fnm0)

    # Initialize trainer
    trainer = Trainer(default_root_dir=checkpoint_dir,
                      accelerator="gpu",
                      devices=1 if torch.cuda.is_available() else None,
                      max_epochs=config_data['max_epochs'],
                      callbacks=[checkpoint_callback],
                      logger=wandb_logger,
                      log_every_n_steps=10
                      )

    # Train the model ⚡
    trainer.fit(model, data_loader)

    # Save
    save_dictionary = config_data
    save_dictionary['model'] = model
    full_checkpoint_path = f"{os.path.dirname(IOdata_fold)}/{checkpoint_fnm0}.ckpt"
    torch.save(save_dictionary, full_checkpoint_path)

    # Evaluate on test set
    # Load model from checkpoint
    state = torch.load(full_checkpoint_path)
    model = state['model']
    trainer.test(model, data_loader)

    # Finalize logging
    wandb.finish()
