from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from baseline_classifier import Baseline


def train(train_dataloader, validation_dataloader, test_dataloader, model, epochs, criterion, optimizer, device): 
    running_train_loss = 0
    validation_loss = 0
    
    model.to(device)
    
    for epoch in range(epochs):
        #Train
        model.train()
        for batch_idx, (X, y) in enumerate(train_dataloader):

            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            preds = model(X)
            preds = preds.argmax(dim=1)
            

            loss = criterion(preds, y)
            loss.backward()
            train_loss = loss.item()
            running_train_loss += train_loss

            optimizer.step()
    
        #Validate
        model.eval()
            
        with torch.no_grad():
            correct_predictions = 0
            total_samples = 0
            for batch_idx, (X, y) in enumerate(validation_dataloader):                

                preds = model(X)
                # Preds are probabilities, so we take the class with the highest probability
                preds = preds.argmax(dim=1)
                loss = criterion(preds, y)
                validation_loss += loss

                correct_predictions += torch.sum(preds.argmax(dim=1) == y).item()
                total_samples += len(y)

        accuracy = correct_predictions / total_samples
        
        print(accuracy)
    
    
    
    #Test
    model.eval()
    
    y_true = np.array([])
    y_pred = np.array([])
    
    with torch.no_grad():
        correct_predictions = 0
        total_samples = 0
        for batch_idx, (X, y) in enumerate(test_dataloader):                

            preds = model(X)
            preds = preds.argmax(dim=1)
            
            y_true = np.concatenate((y_true, y))
            try:
                
                y_pred = np.concatenate((y_pred, preds))
            except ValueError:
                print(preds.shape)
                print(y_pred.shape)
                print(y_pred)
                print(preds) 
                raise ValueError("Error in concatenating predictions.")               
            
    accuracy = accuracy_score(y_pred, y_true)
    
    return accuracy


def kfold_validation(X, y, model, epochs, criterion, optimizer, device, k_folds : int = 5, batch_size : int = 64):
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)
    
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(train_idx.shape)
        train_idx, val_idx = train_test_split(train_idx, train_size=0.8, random_state=0, shuffle=True)

        
        dataset = TensorDataset(X, y)
        
        # Define the data loaders for the current fold
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        
        validation_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        )
        
        test_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )
        
        accuracy = train(train_dataloader, validation_dataloader, test_dataloader, model, epochs, criterion, optimizer, device)

        accuracies.append(accuracy)
    
    
            
    

if __name__ == "__main__":
    print("Train module.")
    
    X = torch.rand(133, 1536, 4)
    y = torch.randint(low=0, high=5, size=(133, ))
    model = Baseline(in_features=4 * 1536, n_classes=5)
    
    print(X.shape)
    print(y.shape)
    
    
    
    kfold_validation(X, y, model, 10, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters()), torch.device("cpu"))
    