import sys
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import HuberLoss
torch.set_default_dtype(torch.float64)


def normalize(data, mean_arr, std_arr):
    nvars = data.shape[1]
    norm_data = torch.zeros_like(data)
    
    for k in range(nvars): # unnorm_data shape = nexamples by nparams
        norm_data[:, k] = (data[:, k] - mean_arr[k]) / std_arr[k]
    return norm_data

def unnormalize(data, mean_arr, std_arr):
    nvars = data.shape[1]
    unnorm_data = torch.zeros_like(data)
    
    for k in range(nvars): # unnorm_data shape = nexamples by nparams
        unnorm_data[:, k] = (data[:, k] * std_arr[k]) + mean_arr[k]
    return unnorm_data


class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_hidden_layers=8, n_hidden_neurons=256):
        super().__init__()
        # Activation function
        self.activation = nn.ReLU()
        # Input layer
        self.input_layer = nn.Sequential(*[nn.Linear(n_input, n_hidden_neurons), self.activation])
        # Individual hidden layer template
        hidden_layer = nn.Sequential(*[nn.Linear(n_hidden_neurons, n_hidden_neurons), self.activation])
        # Hidden layers
        self.hidden_layers = nn.Sequential(*[hidden_layer for _ in range(n_hidden_layers)])
        # Output layer
        self.output_layer = nn.Linear(n_hidden_neurons, n_output)

    # Forward pass through neural network
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


class PCAModel(LightningModule):

    def __init__(self, n_input, n_output, 

                io_stats,               
                key_add_noise_to_Idata=0,
                noise_level=1e-3,

                n_hidden_layers=8, n_hidden_neurons=256, 
                lr=1.e-4):

        super().__init__()
        # self.norm = norm
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.lr = lr

        self.io_stats = io_stats # Dict: mean, std of IOdata
        self.key_add_noise_to_Idata=key_add_noise_to_Idata
        self.noise_level=noise_level   

        # Architecture
        self.model = MLP(n_input, n_output, \
                         n_hidden_layers=n_hidden_layers, \
                         n_hidden_neurons=n_hidden_neurons)
        
        # Loss function
        self.loss_func = HuberLoss()  # consider MSE

    def forward(self, x):
        return self.model.forward(x)

    def forward_unnormalize(self, x):
        return unnormalize(self.forward(x), 
                              self.io_stats['mean_Odata'], 
                              self.io_stats['std_Odata'])

    def add_noise(self, data, noise_level):
        noise = torch.randn_like(data) * noise_level
        return data + noise

    def training_step(self, batch, batch_nb):
        x, y = batch
        
        if self.key_add_noise_to_Idata ==1:
            x = self.add_noise(x, noise_level=self.noise_level)

        y_pred = self.forward(x)
        loss   = self.loss_func(y_pred, y)

        y      = unnormalize(y,      self.io_stats['mean_Odata'], self.io_stats['std_Odata'])
        y_pred = unnormalize(y_pred, self.io_stats['mean_Odata'], self.io_stats['std_Odata'])

        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        mae = torch.abs(y - y_pred)

        # Logging
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_RAE", rae.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_MAE", mae.mean(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch

        if self.key_add_noise_to_Idata ==1:
            x = self.add_noise(x, noise_level=self.noise_level)

        y_pred = self.forward(x)
        loss   = self.loss_func(y_pred, y)

        y      = unnormalize(y,      self.io_stats['mean_Odata'], self.io_stats['std_Odata'])
        y_pred = unnormalize(y_pred, self.io_stats['mean_Odata'], self.io_stats['std_Odata'])

        # computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred)
        av_mae = mae.mean()
        av_mae_wl = mae.mean(0)

        # Logging
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", av_mae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_MAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_mae_wl)]
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_RAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_rae_wl)]
        self.log("valid_CorrCoeff", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch

        if self.key_add_noise_to_Idata ==1:
            x = self.add_noise(x, noise_level=self.noise_level)

        y_pred = self.forward(x)
        loss   = self.loss_func(y_pred, y)

        y      = unnormalize(y,      self.io_stats['mean_Odata'], self.io_stats['std_Odata'])
        y_pred = unnormalize(y_pred, self.io_stats['mean_Odata'], self.io_stats['std_Odata'])

        # computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred)
        av_mae = mae.mean()
        av_mae_wl = mae.mean(0)

        # Logging
        # Why following still say "valid" and not "test"
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", av_mae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_MAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_mae_wl)]
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_RAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_rae_wl)]
        self.log("valid_CorrCoeff", cc, on_epoch=True, prog_bar=True, logger=True)
        
        # self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_MAE", av_mae, on_epoch=True, prog_bar=True, logger=True)
        # [self.log(f"test_MAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_mae_wl)]
        # self.log("test_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        # [self.log(f"test_RAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_rae_wl)]
        # self.log("test_CorrCoeff", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
