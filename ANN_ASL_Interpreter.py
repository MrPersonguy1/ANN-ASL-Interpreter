# -*- coding: utf-8 -*-
"""
Import Statements
"""

import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm

"""
#Import the Dataset
"""

num_to_letter = {'0':'A', '1':'B', '2':'C', '3':'D', '4':'E', '5':'F', '6':'G', '7':'H', '8':'I', '9':'K', '10':'L', '11':'M', '12':'N', '13':'O', '14':'P', '15':'Q', '16':'R', '17':'S', '18':'T', '19':'U', '20':'V', '21':'W', '22':'X', '23':'Y', '24':'Z', }

"""#### Import Kaggle & the Dataset"""

!pip install --upgrade kagglehub

import kagglehub

path = kagglehub.dataset_download("datamunge/sign-language-mnist")
print("Path to dataset files:", path)

print(os.listdir(path))

train_dir = os.path.join(path, "sign_mnist_train.csv")
test_dir = os.path.join(path, "sign_mnist_test.csv")

train_df=pd.read_csv(train_dir)
test_df=pd.read_csv(test_dir)

train_images=train_df.drop('label',axis=1).values.reshape((-1,28,28,1))
train_labels=train_df['label'].values

test_images=test_df.drop('label',axis=1).values.reshape((-1,28,28,1))
test_labels=test_df['label'].values

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

train_images[0].shape

"""
#Visualize the Data
"""

image_num = 1000 #@param {type:"raw"}

print(train_images[image_num].shape)

plt.imshow(train_images[image_num])

np.unique(train_labels)

"""#Create the Model"""

class MLP(nn.Module):
  def __init__(self, input_layer=784, output=25):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_layer, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, output)
    )

  def forward(self, x):
    return self.layers(x)

model = MLP()

from torchsummary import summary

summary(model, input_size=(1, 28, 28))

"""#Create the dataloaders"""

train_images = torch.tensor(train_images, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_images = torch.tensor(test_images, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

train_set = torch.utils.data.TensorDataset(train_images, train_labels)
test_set = torch.utils.data.TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_images, batch_size=32, shuffle=False)
test_loader = DataLoader(test_images, batch_size=32, shuffle=False)

"""
#Create the Loss Function & Optimizer
"""

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

"""
#Create the Train Loop
"""

def train_loop(model, train_loader, loss_fn, optimizer, epochs):
  train_loss = []
  model.train()

  for epoch in range(epochs):
    train_loss_epoch = 0

    for image, label in tqdm(train_loader, desc="Training Model"):
      pred = model(image)
      loss = loss_fn(pred, label)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_loss_epoch += loss.item()

    train_loss.append(train_loss_epoch / len(train_loader))
    print(f'Epoch: {epoch+1} | Loss: {avg_loss:.4f}')

  return train_loss

losses = train_loop(model, train_loader, loss_fn, optimizer, epochs=10)

"""#Visualize the Loss Drop"""

epoch_list = list(range(1, 11))
plt.plot(epoch_list, losses)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

"""#Creating the Testing Function"""

def accuracy(correct, total):
  return correct/total * 100

def test_loop(test_dataloader, model):
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for image, label in tqdm(test_dataloader, desc="Testing Model"):
      pred = model(image)
      correct += (pred.argmax(1) == label).type(torch.float).sum().item()
      total += len(label)

    print(f'Correct: {correct} / Total: {total}')

  accuracy = accuracy(correct, total)

  return accuracy

accuracy = test_loop(test_loader, model)

plt.imshow(test_images[1].squeeze())
plt.title(f"Label: {test_labels[1]}")
plt.axis("off")
plt.show()

with torch.no_grad():
  pred = model(test_images[image].to(torch.float32).unsqueeze(0)) # Add a dimension for batch
  print(num_to_letter[f"{pred.argmax(1).item()}"])
