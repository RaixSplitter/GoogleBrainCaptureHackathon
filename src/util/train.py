from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch

def train(train_dataloader, validate_dataloader, test_dataloader, model, epochs, criterion, optimizer, device): 
    running_train_loss = 0
    validation_loss = 0
    
    model.to(device)
    
    for epoch in range(epochs):
        #Train
        model.train()
        for batch_idx, (X, y) in enumerate(train_dataloader):

            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            preds = model(images)

            loss = criterion(preds, labels)
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

                preds = model(images)

                loss = criterion(preds, labels)
                validation_loss += loss

                correct_predictions += torch.sum(preds.argmax(dim=1) == labels).item()
                total_samples += len(labels)

        accuracy = correct_predictions / total_samples
    
    
    
    #Test
    model.eval()
    
    y_true = np.array([])
    y_pred = np.array([])
    
    with torch.no_grad():
        correct_predictions = 0
        total_samples = 0
        for batch_idx, (X, y) in enumerate(validation_dataloader):                

            preds = model(images)
            
            y_true = numpy.concatenate((y_true, y), axis=0)
            y_pred = numpy.concatenate((y_pred, preds), axis=0)
            
    accuracy = accuracy_score(y_pred, y_true)
    
    return accuracy


def kfold_validation(k_folds : int = 5, batch_size : int = 64, X, y):
    
    kf = StratifiedKFold(n_splits=k_folds)
    
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.get_n_splits(X, y)):
        
        train_idx, val_idx = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True, stratify=y)
    
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
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )
        
        test_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )
        
        accuracy = train(train_dataloader, validate_dataloader, test_dataloader, model, epochs, criterion, optimizer, device)

        accuracies.append(accuracy)
    
    
            
    

            