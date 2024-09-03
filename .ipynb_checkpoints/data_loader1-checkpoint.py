import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import torch
import pickle
torch.set_default_dtype(torch.float64)

class PCADataModule(pl.LightningDataModule):

    def __init__(self, data_fnm, seed, 
                 
                 key_add_noise_to_Idata=0,
                 noise_level=1e-3,
                 
                 nb_train=.6, nb_val=.2, nb_test=.2,
                 batch_size=1024, num_workers=8, **kwargs):
        """ Loads paired data samples of AIA EUV images and EVE irradiance measures.

        Input data needs to be paired.
        Parameters
        ----------
        stacks_csv_path: path to the matches
        eve_npy_path: path to the EVE data file
        """
        super().__init__()
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2
        self.batch_size = batch_size
        self.nb_train = nb_train
        self.nb_val = nb_val
        self.nb_test = nb_test
        # self.data_path = data_path
        self.data_fnm = data_fnm # PA
        self.io_stats = None # PA
        
        self.key_add_noise_to_Idata=key_add_noise_to_Idata
        self.noise_level=noise_level       
        self.seed = seed

    # ------------------------------------------------------------------------
    def compute_IOdata_stats(self, input_data, output_data):
        # compute mean and standard deviation of each input/output parameter
        mean_Idata = np.mean(input_data, axis=0)
        std_Idata  = np.std(input_data, axis=0)
        
        mean_Odata = np.mean(output_data, axis=0)
        std_Odata  = np.std(output_data, axis=0)

        IOdata_stats_dict = dict(mean_Idata=mean_Idata, std_Idata=std_Idata,
                                 mean_Odata=mean_Odata, std_Odata=std_Odata)

        # write stats to dict/file
        stats_fnm1 = self.data_fnm.split('.')[0]+ '_IOstats.pickle'
        with open(stats_fnm1, 'wb') as f:
            pickle.dump(IOdata_stats_dict, f)

        return IOdata_stats_dict

    # ----------------------------------
    def normalize_data(self, data, mean_arr, std_arr):
        nvars = data.shape[1]
        norm_data = np.zeros_like(data)

        for k in range(nvars):
            norm_data[:,k] = (data[:,k] - mean_arr[k])/std_arr[k]

        return norm_data

    def add_noise(self, data, noise_level):
        noise = np.random.randn(*data.shape)*noise_level
        return data + noise

    # ----------------------------------
    def setup(self, stage=None):
        
        # load data
        with open(self.data_fnm, "rb") as f:
            tdict = pickle.load(f)
        input_data  = tdict['input_data']
        output_data = tdict['output_data']

        # -------------
        # create temporary noisy data Input data - for IOstats
        input_data1 = input_data+0.
        if self.key_add_noise_to_Idata ==1: # then create noise data, add to Idata and compute mean/std for this noisy data
            void = np.random.seed(seed=None)
            input_data1 = self.add_noise(input_data1, self.noise_level)
        input_data1[:,-1] = input_data[:,-1]+0 # store projH as is
            
        # Get IOdata stats --------
        IOdata_stats_dict = self.compute_IOdata_stats(input_data1, output_data)
        self.io_stats = IOdata_stats_dict
        input_data1 =None # free memory

        # normalize IO data --------
        input_data  = self.normalize_data(input_data, \
                                          self.io_stats['mean_Idata'], \
                                          self.io_stats['std_Idata'])
        output_data = self.normalize_data(output_data, \
                                          self.io_stats['mean_Odata'], \
                                          self.io_stats['std_Odata'])

        # extract 1st 10k models for evaluation dataset --------
        input_data  =  input_data[10000:,:] 
        output_data = output_data[10000:,:] 

        # --------------------------------------------------
        generator1 = torch.Generator().manual_seed(self.seed)

        self.train_sampler, self.val_sampler, self.test_sampler = \
                    random_split(range(input_data.shape[0]), \
                    [self.nb_train, self.nb_val, self.nb_test], \
                    generator=generator1)
        
        self.dataset = dataFn(input_data, output_data)
        
    # --------------------------------------------------
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, \
                          num_workers=self.num_workers, \
                          sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, \
                          num_workers=self.num_workers, \
                          sampler=self.val_sampler)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, \
                          num_workers=self.num_workers, \
                          sampler=self.test_sampler)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, \
                          num_workers=self.num_workers, \
                          sampler=self.val_sampler)

class dataFn(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1[:,0])

    def __getitem__(self, idx):
        return self.data1[idx,:], self.data2[idx,:]
