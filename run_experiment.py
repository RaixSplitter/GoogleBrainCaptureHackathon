import torch
import numpy as np

from model.train import kfold_validation
from model.baseline_classifier import Baseline
from model import BendrEncoder
from copy import deepcopy
from latent_encoding import generate_latent_representations


print("Train module.")
    
X = torch.load("X_data.pt")
y = torch.load("y_data.pt")

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
encoder = BendrEncoder()

# Load the pretrained model
encoder.load_state_dict(deepcopy(torch.load("encoder.pt", map_location=device)))
encoder = encoder.to(device)


model = Baseline

cfg = {
    "tqdm_disabled" : False,
    "epoch" : 3,
    "k_folds" : 5,
    "batch_size" : 64
}



accuracies, stats = kfold_validation(X, y, model, encoder, cfg)

import pickle

for key, value in stats.items():
    print(f"Key: {key}")
    print(f"Value: {np.array(value)}")
    
with open('stats.pickle', 'wb') as f:
    pickle.dump(stats, f)