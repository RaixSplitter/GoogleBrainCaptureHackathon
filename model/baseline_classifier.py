import torch
import torch.nn as nn



class Baseline(nn.Module):
    """
    Simple CNN network for doing classification on 24 classes

    ARGS: 
        in_features (int): number of input features
        n_classes (int): number of classes to predict
    
    RETURNS:
        model (nn.Module): a PyTorch model
    """

    def __init__(self, in_features:int, n_classes: int, encoder) -> None:
        """
        A small network for doing classification   
        The architecture of the network is adapted from: https://www.kaggle.com/code/lightyagami26/mnist-sign-language-cnn-99-40-accuracy
        """
        super(Baseline, self).__init__()
        self.in_features = in_features
        self.n_classes = n_classes

        self.prep_block = nn.Sequential(
            nn.Flatten() # out: 4 * 1536
        )
        self.dense_block = nn.Sequential(
            encoder,
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=512*4),
            nn.Dropout(p=0.4, inplace=False),
            nn.BatchNorm1d(512 * 4),
            nn.ReLU(),
            nn.Linear(in_features=512*4, out_features=n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x) -> torch.Tensor:
        """ Forward pass """
        x = self.dense_block(x)
        return x
    
