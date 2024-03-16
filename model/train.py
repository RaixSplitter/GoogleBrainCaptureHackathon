from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from torch import nn

    


def train(train_dataloader, validation_dataloader, test_dataloader, model, criterion, optimizer, device, cfg): 
    running_train_loss = 0
    validation_loss = 0
    test_accuracy = 0
    train_val_accuracies = []
    epochs = cfg["epoch"]
    model.to(device)
    
    stats = {
        "train_loss" : [],
        "avg_train_loss" : [],
        "val_loss" : [],
        "val_acc" : [],
    }
    
    pbar = tqdm(range(epochs), disable = cfg["tqdm_disabled"])
    for epoch in pbar:
        stats["train_loss"].append([])
            
        #Train
        model.train()
        
        for batch_idx, (X, y) in enumerate(train_dataloader):

            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            propabilities = model(X)
            preds = propabilities.argmax(dim=1)
            

            loss = criterion(propabilities, y)
            loss.backward()
            train_loss = loss.item()
            
            
            
            running_train_loss += train_loss

            optimizer.step()
            
            #Train loss
            stats["train_loss"][-1].append(train_loss)
    
        #Validate
        model.eval()
            
        with torch.no_grad():
            correct_predictions = 0
            total_samples = 0
            for batch_idx, (X, y) in enumerate(validation_dataloader):                

                propabilities = model(X)
                preds = propabilities.argmax(dim=1)
                loss = criterion(propabilities, y)
                validation_loss += loss

                correct_predictions += torch.sum(preds == y).item()
                total_samples += len(y)

        train_val_acc = correct_predictions / total_samples
        train_val_accuracies.append(train_val_acc)
        avg_loss = running_train_loss / (epoch + 1)
        avg_acc = np.mean(train_val_accuracies)
        pbar.set_description(f"Epoch: {epoch+1}/{epochs} avg trainingloss : {avg_loss}, avg_val_acc : {avg_acc}")
        
        
        
        
        #Avg Train loss
        stats["avg_train_loss"].append(avg_loss)
        #Val loss
        stats["val_loss"].append(validation_loss)
        
        #Val accuracy
        stats["val_acc"].append(train_val_acc)
        
        
    
    
    
    #Test
    model.eval()
    y_true = np.array([])
    y_pred = np.array([])
    
    with torch.no_grad():
        correct_predictions = 0
        total_samples = 0
        for batch_idx, (X, y) in enumerate(test_dataloader):                

            propabilities = model(X)
            preds = propabilities.argmax(dim=1)
            
            y_true = np.concatenate((y_true, y))

            y_pred = np.concatenate((y_pred, preds))
                       
            
    test_accuracy = accuracy_score(y_pred, y_true)
    
    
    
    print(f"test_acc : {test_accuracy}")
    
    return test_accuracy, stats


def kfold_validation(X, y, model, encoder, cfg):
    
    kf = KFold(n_splits=cfg["k_folds"], shuffle=True, random_state=0)
    
    accuracies = []
    
    stats = {
        "train_loss" : [],
        "avg_train_loss" : [],
        "val_loss" : [],
        "val_acc" : [],
    }
    
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        for value in stats.values():
            value.append([])
        
        
        train_idx, val_idx = train_test_split(train_idx, train_size=0.8, random_state=0, shuffle=True)

        
        dataset = TensorDataset(X, y)
        
        # Define the data loaders for the current fold
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg["batch_size"],
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        
        validation_dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg["batch_size"],
            sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        )
        
        test_dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg["batch_size"],
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )
        
        model_instance = model(in_features=4 * 1536, n_classes=5, encoder=encoder)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_instance.parameters())
        device = torch.device("cpu")
        
        accuracy, train_stats = train(train_dataloader, validation_dataloader, test_dataloader, model_instance, criterion, optimizer, device, cfg)
        
        for key, value in train_stats.items():
            stats[key][-1].append(value)
            
        
        
        accuracies.append(accuracy)
        
    stats["test_acc"] = accuracies
    
    return accuracies, stats
    
    
    
            
    
