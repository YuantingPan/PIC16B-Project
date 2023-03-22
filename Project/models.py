import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from tqdm import tqdm


def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    """
    Train the MLP classifier on the training set and evaluate it on the validation set every opch
    
    Args:
        model (MLP): MLP classifier to train.
        train_loader (torch.utils.data.DataLoader): Data loader for the training set.
        val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        criterion (callable): Loss function to use for training.
        device (torch.device): Device to use for training.
        num_epochs (int): Number of epochs to train the model.
    """
    
    # Place model on device
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        
        # Use tqdm to displace a progress bar during training
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for inputs, labels in train_loader:
                # Move inputs and labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero out gradients
                optimizer.zero_grad()
                
                # Compute the logits and loss
                logits = model(inputs)
                loss = criterion(logits, labels)
                
                # Backpropagate the loss
                loss.backward()
                
                # Update the weights
                optimizer.step()
                
                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                
        # Evaluate the model on the validation set
        total_loss, avg_loss, accuracy = evaluate(model, val_loader, criterion, device)
        print(f'Validation set: Total loss = {total_loss:.4f}, Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the MLP classifier on the test set.
    
    Args:
        model (MLP): MLP classifier to evaluate.
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        criterion (callable): Loss function to use for evaluation.
        device (torch.device): Device to use for evaluation.
        
    Returns:
        float: Average loss on the test set.
        float: Accuracy on the test set.
    """
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0
        
        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Compute the logits and loss
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Compute the accuracy
            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)
            
    # Compute the average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples
    
    return total_loss, avg_loss, accuracy


# List of image transformation
image_blurrer = transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.001,0.2))
image_rotate = transforms.RandomRotation((-5, 5))
image_hflip = transforms.RandomHorizontalFlip(p=0.2)
image_vflip = transforms.RandomVerticalFlip(p=0.4)
image_randomP = transforms.RandomPerspective(distortion_scale=0.1, p=0.25)


class Resnet(nn.Module):
    def __init__(self, mode='finetune', augmented=False, pretrained=True):
        super().__init__()
        """
        use the resnet18 model from torchvision models. Remember to set pretrained as true
        
        mode has two options:
        1) linear: For this mode, we want to freeze resnet18 feautres, then train a linear
                    classifier which takes the features before FC (again we do not want the resnet18 FC).
                    And then write our own FC layer: which takes in the features and 
                    output scores of size 7 (since we have 7 categories)
        2) finetune: Same as 1) linear, except that we do not need to freeze the features and
                    can finetune on the pretrained resnet model.
        """
    
            
        self.augmented = augmented
        self.resnet = models.resnet50(pretrained=True)

        if mode == 'linear':
            for name, param in self.resnet.named_parameters():
                if param.requires_grad and 'fc' not in name:
                    param.requires_grad = False
            
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_features, 7)

        elif mode == 'finetune':
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_features, 7)

    def forward(self, x):
        if self.augmented:
            x = image_hflip(x)
            x = image_rotate(x)
            x = image_randomP(x)
        
        x = self.resnet(x)
        return x

    def to(self, device):
        return self.resnet.to(device=device)






























