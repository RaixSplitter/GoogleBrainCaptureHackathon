from model import BendrEncoder
import torch
from copy import deepcopy

import numpy as np
from tqdm import tqdm

def generate_latent_representations(data, encoder, batch_size=5):
    """ Generate latent representations for the given data using the given encoder.
    Args:
        data (np.ndarray): The data to be encoded.
        encoder (nn.Module): The encoder to be used.
        batch_size (int): The batch size to be used.
    Returns:
        np.ndarray: The latent representations of the given data.
    """
    latent_size = (1536, 4) # do not change this 
    latent = np.empty((data.shape[0], *latent_size))

    for i in tqdm(range(0, data.shape[0], batch_size)):
        latent[i:i+batch_size] = encoder(data[i:i+batch_size]).cpu().detach().numpy()
    
    return latent.reshape((latent.shape[0], -1))