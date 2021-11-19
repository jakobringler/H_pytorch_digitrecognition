# hou

node = hou.pwd()
geo = node.geometry()

# import modules

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import jit

from torch.utils import backcompat
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.transforms as transforms

# load on cpu config

device = torch.device('cpu')

# hyper parameters

input_size = 784 # 28 x 28 pixels
hidden_size = 100
num_classes = 10 # 10 digits 
num_epochs = 25
batch_size = 100
learning_rate = 0.001

# import all points

N = len(geo.points())

# create numpy array from input point attribute array

numpy_input = np.zeros((N,784), 'float32')
numpy_output =  np.zeros((N,1), 'float32')

for i,point in enumerate(geo.points()):
    numpy_input[i] = np.asarray(point.attribValue('input'))
    
class digitDataset(Dataset):
    def __init__(self, transform=None):
        self.n_samples = numpy_input.shape[0]

        # note that we do not convert to tensor here
        self.input = torch.from_numpy(numpy_input)

        self.transform = transform

    def __getitem__(self, index):
        sample = self.input[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples
        
train_dataset = digitDataset()

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    
#print(numpy_input)

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

model = NeuralNet(input_size, hidden_size, num_classes)

# import model

PATH = "`$HIP`/model.pth"

model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()
    
# predict output with model data

#print(model.state_dict())

with torch.no_grad():
    target = model(torch.from_numpy(numpy_input[0]))
    
print(target)

#target = net(input).detach().numpy()
'''  
# set point attributes

for i,point in enumerate(geo.points()):
    point.setAttribValue("target",np.absolute(target[i].astype(np.float64)))
'''