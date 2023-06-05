#filtering and feature extraction of acceleration data
import numpy as np
from numpy import genfromtxt

data = genfromtxt('ACCtest.csv', delimiter=',')
data = data[1:,:] #remove the header row

#extract the mean from each axis. Order: x1, y1, z1, 3D1, x2, y2, z2, 3D2. 
#3D refers to the magnitude of movement

acc_data = data[:,4:7] #takes the first acceleration data columns from the file
acc_data = np.concatenate((acc_data, np.sqrt(np.sum((acc_data)**2, axis=1)[:,np.newaxis])), axis=1) 

acc_data2 = data[:,7:10] #takes the next three acceleration data columns from the file
acc_data = np.concatenate((acc_data, acc_data2, np.sqrt(np.sum((acc_data2)**2, axis=1)[:,np.newaxis])), axis=1) 

print(acc_data[0])

means = np.mean(acc_data, axis=0)
stds = np.std(acc_data, axis=0)
peakVals = np.max(acc_data, axis=0)

print(means)
print(stds)
print(peakVals)
