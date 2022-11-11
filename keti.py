# -*- coding: utf-8 -*-
"""
Created on Thu Nov 6

@author: Huang
"""






#train funt
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#test func
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
#model loader
def modelloader():
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))


#dataloader
def dataloader():
    batch_size = 64
    
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)


#data
def data():
    #train data
    roottrain = input("train root:")
    training_data = datasets.FashionMNIST(
        root=roottrain,
        train=True,
        download=False,
        transform=ToTensor()
    )

    #test data
    roottest = input("train root:")
    test_data = datasets.FashionMNIST(
        root=roottest,
        train=False,
        download=False,
        transform=ToTensor()
    )








import torch
import os
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
#customimagedataset
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
model = NeuralNetwork().to(device)
print(model)


"""
#train data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

#test data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
"""





#add gui






#mode
mode = int(input("mode(1 for training,2 for using):"))
if mode == 1:
    
    havedata = int(input("data already(1 for true,2 for false):"))
    if havedata == 1:
        data()
        
    elif havedata == 2:
        filedir = input("data dir:")
        labledir = input("lable file dir:")
        CustomImageDataset.__init__("",labledir,filedir)
    #dataloader()
    batch_size = 64
    
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    #modelloader()
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
         
    #loss funt
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


    train(train_dataloader,model,loss_fn,optimizer)

    test(dataloader, model, loss_fn)







    #save model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
elif mode == 2:
    
    havedata = int(input("data already(1 for true,2 for false):"))
    if havedata == 1:
        data()
        dataloader()
    elif havedata == 2:
        filedir = input("data dir:")
        labledir = input("lable file dir:")
        CustomImageDataset.__init__("",labledir,filedir)
        batch_size = 64
        
        # Create data loaders.
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
        
        #modelloader
        model = NeuralNetwork()
        model.load_state_dict(torch.load("model.pth"))




    
    
    
    









    
