node = hou.pwd()
geo = node.geometry()

# Add code to modify contents of geo.
# Use drop down menu to select examples.

import numpy as np
import torch
from datetime import datetime

allPoints = geo.points()

N = len(allPoints) # samples

h = 28
w = h
c = 1

data = np.zeros((w,h,c), dtype=np.uint8) # this has to be the same as before
shape = data.shape

numpy_input = np.zeros((N,784), 'float32')

for i,point in enumerate(allPoints):    

    numpy_input[i] = np.asarray(point.attribValue('input'))
    input_np = np.asarray(numpy_input.astype(np.uint8)).reshape(shape)
    input_np = np.swapaxes(input_np,1,2)
    input_np = np.swapaxes(input_np,0,1)
    input_np = np.swapaxes(input_np,1,2)
    input = torch.from_numpy(input_np)
    print(input)
    print(input.shape)

print(datetime.now())