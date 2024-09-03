import os
import torch
from torch.utils.data import DataLoader

def ipredict(model, dataset, return_inputs=False, batch_size=2, num_workers = None):
    """Predict inversion result for a given set of input PCA parameters.

    Parameters
    ----------
    model: torch.nn.Module, to be loaded prior to here
    dataset: pytorch dataset for streaming the input data.
    return_inputs: set True to return input images.
    batch_size: number of samples to process in parallel.
    num_workers: number of workers for data preprocessing (default cpu_count / 2).

    Returns
    -------
    predicted irradiance as numpy array and corresponding image if return_images==True
    """

    # use model after training or load weights and drop into the production system
    model.eval()
    # load data
    num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(model.device)
            # pred_irradiance = imgs
            # pred_irradiance = model.forward_unnormalize(imgs)
            pred_irradiance = model.forward_unnormalize(imgs) # works on output data only
            
            for pred, img in zip(pred_irradiance, imgs):
                if return_inputs:
                    yield pred.cpu(), img.cpu()
                else:
                    yield pred.cpu()

