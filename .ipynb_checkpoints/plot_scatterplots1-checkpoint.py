import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch
from tqdm import tqdm
import pandas as pd
import datetime
import matplotlib.dates as mdates
import dateutil.parser as dt
from codes.infer1 import ipredict
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

def normalize(data, mean_arr, std_arr):
    nvars = data.shape[1]
    norm_data = np.zeros_like(data)
    
    for k in range(nvars): # unnorm_data shape = nexamples by nparams
        norm_data[:, k] = (data[:, k] - mean_arr[k]) / std_arr[k]
    return norm_data

def add_noise(data, noise_level):
    noise = np.random.randn(*data.shape) * noise_level
    return data+noise

def pearson_CC(var1_arr=None, var2_arr=None):
    # set values that are inf to nan
    var1_arr[np.isinf(var1_arr)] = np.nan
    var2_arr[np.isinf(var2_arr)] = np.nan

    # set values that are NaN in the 2nd array to be NaN in the 1st array and vice, versa
    var1_arr[np.isnan(var2_arr)] = np.nan
    var2_arr[np.isnan(var1_arr)] = np.nan


    mean_var1 = np.nanmean(var1_arr)
    mean_var2 = np.nanmean(var2_arr)
    
    tpar1 = np.sqrt(np.nansum((var1_arr - mean_var1)**2)) # sqrt of sum of square difference
    tpar2 = np.sqrt(np.nansum((var2_arr - mean_var2)**2)) # sqrt of sum of square difference

    
    # print(mean_var1, mean_var2, tpar1, tpar2)

    tpar12 = np.nansum((var1_arr - mean_var1) * (var2_arr - mean_var2))
    
    
    return tpar12/(tpar1*tpar2) # pearson correlation
    

# ----------------------------------------------------------
if __name__ == "__main__":
    # -----------
    # work_dir = '/glade/u/home/piyushag/cmex_ml0'
    # tdir = 'NN_IOdata_ckpts_Jul11onwards'
    work_dir = '/glade/work/piyushag/cmex_ml0'
    tdir = 'NN_IOdata_ckpts_Jul26onwards'
    
    # --------------------------
    # IOdata_fnm0 = 'NN_IOdata_MgIIhk_PCAdbase_nPSF_may17_1st10k_MgIIhk_Obs_nPSF_Idata=coeffs_pH=1_absB=1_nl=1e-03.pickle'
    # IOdata_fnm00 = IOdata_fnm0.split('.')[0]
    # checkpoint_fnm0 = f'{IOdata_fnm00}_nlCoeff=0_maxEpoch=100'
    # IOdata_fnm0 = 'NN_IOdata_Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_Idata=coeffs_pH=1_absB=1_nl=1e-03.pickle'
    # IOdata_fnm00 = 'Jul26_pcaDB_LHS_2M_Blin_1st10k_MgIIhk_Obs_nPSF_nl=1e-03_Dsize=1M_pH=1_Btan=1_hcosTh=1'
    # IOdata_fnm00 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_nl=1e-03_Dsize=1M_pH=1_Btan=1_hsinTh=1'
    # IOdata_fnm00 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_Idata=coeffs_nl=1e-03_Dsize=1M_pH=0_absB=1_hcosTh=0'
    # IOdata_fnm00 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_Idata=coeffs_nl=1e-03_Dsize=1M_pH=1_Btan=1_hcosTh=1'
    # IOdata_fnm00 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_Idata=coeffs_nl=1e-03_Dsize=1M_pH=0_Btan=1_hcosTh=0'

    # IOdata_fnm00 = 'MgIIhk_PCAdbase_nPSF_may17_1st10k_MgIIhk_Obs_nPSF_nC=9532_nl=1e-03_Dsize=1M_pH=1_Btan=1_hsinTh=1'
    # IOdata_fnm00 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_nC=9532_nl=1e-03_Dsize=1M_pH=1_Btan=1_habscosTh=1'
    IOdata_fnm00 = 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_nC=9532_nl=1e-03_Dsize=4M_pH=1_Btan=1_habscosTh=1'
    
    
    IOdata_fnm0 = f'{IOdata_fnm00}.pickle'
    checkpoint_fnm0 = f'{IOdata_fnm00}_nlC=0_maxEpoch=200'
    
    # 'Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_Idata=coeffs_nl=1e-03_Dsize=1M_pH=0_absB=1_hcosTh=0_nlC=0_maxEpoch=50.ckpt'
    # IOdata_fnm0 = 'NN_IOdata_MgIIhk_PCAdbase_nPSF_may17_1st10k_MgIIhk_Obs_nPSF_Idata=coeffs_pH=1_absB=1_nl=0.pickle'
    # IOdata_fnm00 = IOdata_fnm0.split('.')[0]
    # checkpoint_fnm0 = 'NN_IOdata_Jul5_pcaDB_LHS_5M_1st10k_MgIIhk_Obs_nPSF_Idata=coeffs_pH=1_absB=1_nl=1e-03_nlCoeff=0_5M_maxEpoch=100'
    # checkpoint_fnm0 = f'{IOdata_fnm00}_nlCoeff=1e-03_maxEpoch=100'

    
    IOdata_fnm = f"{work_dir}/{tdir}/NN_IOdata/{IOdata_fnm0}"
    IOstats_fnm= '{}_IOstats.pickle'.format(IOdata_fnm.split('.pickle')[0])
    checkpoint_fnm = f"{work_dir}/{tdir}/{checkpoint_fnm0}.ckpt"

    result_path = f"{work_dir}/{tdir}/results"
    os.makedirs(result_path, exist_ok=True)

    
    # read IOdata --------------------
    with open(IOdata_fnm, "rb") as f:
        tdict = pickle.load(f)
    input_data  = tdict['input_data'][:10000,:]
    output_data = tdict['output_data'][:10000,:]
    input_data_str  = tdict['input_data_str']
    output_data_str = tdict['output_data_str']

    # read IOstats file -----------------
    with open(IOstats_fnm, "rb") as f:
        IOstats_dict = pickle.load(f)

    print('\n output_data', output_data.shape)
    for k in range(output_data.shape[1]):
        print(output_data_str[k], np.min(output_data[:,k]), np.max(output_data[:,k]))
    
    # add noise to input data -------------------
    # raise NameError('PAerror: Check for consistency in noise realizations for nl=? and nlCoeffs=?')
    # if ((not 'nl=0' in checkpoint_fnm) & (not 'nlCoeff=0' in checkpoint_fnm)):
    #     # atleast one should be 0. You can't pre-put noise to Coeffs and then also add noise during
    #     # the training period.        
    #     raise NameError()    
    # if 'nlCoeff=0' in checkpoint_fnm:
    #     noise_level = np.float64(checkpoint_fnm0.split('_nl=')[1].split('_nlCoeff')[0])
    # else:
    #     noise_level = np.float64(checkpoint_fnm0.split('_nlCoeff=')[1].split('_maxE')[0])
    # input_data1 = add_noise(input_data, noise_level)
    # input_data1[:,-1] = input_data[:,-1]+0.

    # input_data = input_data1+0.
    # input_data1 =None
    
    
    
    # if not 'nl=0' in checkpoint_fnm0:
    #     noise_level = np.float64(checkpoint_fnm0.split('_nl=')[1])
    #     input_data1 = add_noise(input_data, noise_level)
    #     input_data1[:,-1] = input_data[:,-1]+0.

    #     input_data = input_data1+0.
    #     input_data1 =None

    # normalize input data -----------------
    input_data = normalize(input_data, IOstats_dict['mean_Idata'], 
                            IOstats_dict['std_Idata'])


    # ---------------------------------------------------
    # Initalize model
    state = torch.load(checkpoint_fnm)
    model = state['model']

    # Do NN inversion --------------------------
    input_data = (torch.stack([torch.tensor(input_data)]))[0, :, :]
    print(f"shape of input_data: {input_data.shape}")    
    test_inv_result = [irr for irr in tqdm(ipredict(model, input_data),
                                           total=input_data.shape[1])]
    NNinv_data = torch.stack(test_inv_result).numpy()

    print('\n NN res', NNinv_data.shape)
    for k in range(output_data.shape[1]):
        print(output_data_str[k], np.min(NNinv_data[:,k]), np.max(NNinv_data[:,k]))


    # Plot -------------------------
    # nvars = np.arange() 
    fig, axs = plt.subplots(3,3, figsize=(19,11))
    axs = axs.flat

    # k1=-1
    for k in range(output_data.shape[1]): # range(9):
        tparam_str = output_data_str[k]
        var1_arr = output_data[:,k]
        var2_arr = NNinv_data[:,k]        
        
        
        skip_By=0
        if 'Bx' in tparam_str and 'Btan' not in tparam_str:
            k1 = 0

            tkx = k # Bx index
            tky = k+1 # By or absBy index

            if 'By' in output_data_str[tky]:
                var1_arr = np.sqrt(output_data[:,k]**2 + output_data[:,tky]**2) # obsBtan
                var2_arr  = np.sqrt(NNinv_data[:,k]**2 + NNinv_data[:,tky]**2) # NN_Btan
            else:
                print(output_data_str[tky])
                print(output_data_str)
                raise NameError('tky is not the correct index for By')

            tparam_str = 'Btan = √(Bx^2+By^2)'
        elif 'Btan' in tparam_str:
            k1=0
            
        elif 'By' in tparam_str:
            continue
            k1 = 1
        elif 'Bz' in tparam_str:
            k1 = 2
        elif 'T' in tparam_str:
            k1 = 3
        elif 'Vlos' in tparam_str:
            k1 = 4
        elif 'tau' in tparam_str:
            k1 = 5
        elif 'anired' in tparam_str:
            k1 = 6
        elif tparam_str =='h':
            k1 = 7
        elif tparam_str =='costheta':
            k1 = 8
        elif tparam_str =='sintheta':
            k1 = 8
        elif tparam_str =='abscostheta':
            k1 = 8

        tpCC = pearson_CC(var1_arr= var1_arr,
                          var2_arr= var2_arr) # pearson correlation
        
        
        
        # k1+=1; 
        ax =axs[k1]
        ax.plot(var1_arr, var2_arr, '.r', markersize=5)
        ax.set_xlabel("test {}_pCC={:.2f}".format(tparam_str, tpCC))
        ax.set_ylabel("NN {}".format(tparam_str))
        line = np.linspace(var1_arr.min(), var1_arr.max())
        ax.plot(line, line, 'b--')

        # if k in [1,2]:
        #     ax.set_yscale('log')
        #     ax.set_xscale('log')

    # save plot ----------------
    # results_fold = f'{work_dir}/results'
    # os.makedirs(results_fold, exist_ok=True) 
    result_fnm1 = '{}/{}_Invtestmodels'.format(result_path, checkpoint_fnm0)
    # if key_add_noise ==1:
    #     result_fnm1 = '{}/{}_NNinvRes'.format(result_path, checkpoint_fnm0)
    # else:
    #     result_fnm1 = '{}_NNinvRes_nl=0'.format(IOdata_fnm.split('.')[0])

    axs[0].set_title(' {}\n {}'.format(result_path, os.path.basename(checkpoint_fnm0)), loc='left')
    fig.savefig(f'{result_fnm1}.pdf', bbox_inches='tight')
    plt.close()


    # save to file ----------------
    tdict = dict(NNinv_data=NNinv_data, 
                 output_data = output_data, 
                 input_data=input_data, )

    with open(f'{result_fnm1}.pickle', 'wb') as f:
        pickle.dump(tdict, f)
        
  