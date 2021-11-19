node = hou.pwd()
geo = node.geometry()

# Add code to modify contents of geo.
# Use drop down menu to select examples.

h = 28
w = h
c = 1

import numpy as np
from datetime import datetime

allPrims = geo.prims()
allPoints = geo.points()

n = len(allPrims) # samples

data = np.zeros((w,h,c), dtype=np.uint8)
shape = data.shape

for i,prim in enumerate(allPrims):

    col, x, y = prim.attribValue('C'), prim.attribValue('gx'), prim.attribValue('gy')

    data[x,y] = [col]

flatData = data.ravel()

for i,point in enumerate(allPoints):
    point.setAttribValue("input", flatData.astype(np.float64)) # to store any array the datatype has to be float64 

# data2 = np.asarray(flatData).reshape(shape)

#print('------------------------------------------')
#print(data)
#print('------------------------------------------')
#print(flatData)
#print('------------------------------------------')
#print(data2)
#print('------------------------------------------')
#print(shape)
#print(data.shape)
#print(data.size)

#print(flatData.astype(np.float64))
#print(datetime.now())