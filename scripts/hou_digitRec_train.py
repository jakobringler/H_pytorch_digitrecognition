# hou

node = hou.pwd()
geo = node.geometry()

# modules

import numpy as np

import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils import backcompat
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.transforms as transforms

from datetime import datetime

# cuda device config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters

input_size = 784 # 28 x 28 pixels
hidden_size = 100
num_classes = 10 # 10 digits 
num_epochs = 20
batch_size = 100
learning_rate = 0.001
train_split = int(len(geo.points())*0.8)
test_split = int(len(geo.points())*0.2)

# Model Output Location

PATH = "`$HIP`/model/model.pth"

# data shape

h = 28
w = h
c = 1 # grayscale = 1 | rgb = 3

data = np.zeros((h,w,c), dtype=np.uint8) # this has to be the same as before
shape = data.shape

# count points

N = len(geo.points()) # samples

# create numpy array from input and target

numpy_input = np.zeros((N,784), 'float32') # 28 x 28 pixels -> 784 
numpy_output =  np.zeros((N,1), 'float32') # 1D output array 

for i,point in enumerate(geo.points()):
    numpy_input[i] = np.asarray(point.attribValue('input'))
    numpy_output[i] = np.asarray(point.attribValue('target'))


# Transformer

class ReshapeToTensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        inputs, targets = sample
        input_np = np.asarray(inputs.astype(np.uint8)).reshape(shape) # reshape to numpy image dimensions ( X, Y, C )
        input_np = np.swapaxes(input_np,0,1)
        inputs = torchvision.transforms.functional.to_tensor(input_np) # performs axes swap from ( X, Y, C ) to ( C, X, Y) and converts from 0-255 uint8 to 0-1 floats
        targets = torch.from_numpy(targets)
        targets = torch.max(targets, 0)[0]
        targets = targets.type(torch.LongTensor)
        return inputs, targets

transform = ReshapeToTensor()


# Dataset

class digitDataset(Dataset):
    def __init__(self, transform=None):
        self.n_samples = numpy_input.shape[0]
        # conversion to tensor happens in transformer 
        self.input = numpy_input
        self.target = numpy_output
        self.transform = transform

    def __getitem__(self, index):
        sample = self.input[index], self.target[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples
        
dataset = digitDataset(transform=transform)


# Training Sets

train_set, test_set = torch.utils.data.random_split(dataset, [train_split, test_split])

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


# Sanity Check 

examples = iter(train_loader)
example_data, example_targets = examples.next()

print("Example Input:\n", example_data[0])
print('--------------------------')
print("Example Input Shape:   ",example_data[0].shape)
print('--------------------------')
print("Example Target:        ",example_targets[0])
print('--------------------------')
print("Example Target Shape:  ",example_targets[0].shape)
print('--------------------------')
print("Example Target Shape:  ",example_targets[0].type())
print('--------------------------')
print("Train Set Batch Shape: ",example_data.shape)
print('--------------------------')
print("Train Set Length:      ",train_set.__len__())
print('--------------------------')
print("Test Set Length:       ",test_set.__len__())
print('--------------------------')


# Model 

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device) # has to be on gpu

# Loss

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training Loop

print('-------  TRAINING   ------')
print('--------------------------')

n_totalsteps = len(train_loader)

for epoch in range(num_epochs):
    for i, (input, target) in enumerate(train_loader):
        input = input.reshape(-1, 28*28).to(device) # has to be on gpu
        target = target.to(device) # has to be on gpu
        
        # forward
        outputs = model(input)
        loss = criterion(outputs, target) 

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_totalsteps}, loss {loss.item():.4f}')


# Save Model

print('--------------------------')

torch.save(model.state_dict(), PATH)

print("Model saved at: ",PATH)
print('--------------------------')
print('------- MODEL EVAL  ------')
print('--------------------------')


# Test Model

model.eval()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for i, (input, target) in enumerate(test_loader):
        input = input.reshape(-1, 28*28).to(device)
        target = target.to(device)
        outputs = model(input)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += target.size(0)        
        print("Predicted Output:  ",predicted[i],"\nTarget:            ",target[i])
        print('--------------------------')
        n_correct += (predicted == target).sum().item()
        
    acc = 100 * (n_correct / n_samples)

    print(acc,"% Accuracy")

print('--------------------------')
print(datetime.now())