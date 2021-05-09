#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage.measure import block_reduce
from skimage.util import view_as_windows
import random
import copy

# Exercice 1

x = cv2.imread("liftingbody.png",0) #Read image without colors
x = cv2.normalize(x.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) #Convert datatype to add noise easily

N_1,N_2 = x.shape

def generateMatrixUniform(lower,upper,N_1,N_2):
    return np.asarray([[random.choice([lower,upper])for j in range(N_2)] for i in range(N_1) ])

def generateGaussianNoise(mean,std,N,M): #Generate a Gaussian Noise array NxM
    return np.asarray([np.random.normal(mean,std,M) for i in range(N)])


# In[2]:


w = generateMatrixUniform(-1,1,N_1,N_2)

density = 0.5
#Set to 0 half of the pixels randomly to have a 0.5 density
iteration = int((N_1*N_2)* density) # Lenght of the watermarking
tabu = []
while(iteration > 0):
    index1 = random.randint(0, N_1-1)
    index2 = random.randint(0, N_2-1)
    if (index1,index2) not in tabu:
        w[index1][index2] = 0
        tabu.append((index1,index2))
        iteration -= 1


# In[3]:



x = cv2.normalize(x.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX)

# Adding noise after changing range to 0 to 255

u = 0
sigma = 1
z = generateGaussianNoise(u,sigma,N_1,N_2)

y = x + w

v = z + y

print("x: ",x)
print("w: ",w)
print("y: ",y)
print("v: ",v)
print("z: ",z)


# In[4]:


##Exercice 2.1

#Extract

w_non_blind = v - x

#Linear correlation computation beteen w and w_non_blind, only on point which weren't set to 0 in w.

def linearCorrelation(original,estimation,density,tabu):
    row,col = original.shape
    N = int((row*col)* density)
    
    correlation = [estimation[i][j] * original[i][j] for j in range(col) for i in range(row) if (i,j) not in tabu ] 
    
    return sum(correlation)/N

linear_corr = linearCorrelation(w,w_non_blind,density,tabu)


# In[5]:


#padding 0 to have exactly the same size when doing overlapping blocks mean

v_pad = np.zeros((514, 514))
shape = v.shape
v_pad[:shape[0],:shape[1]] = v
print(v_pad.shape)
print(v_pad)


# In[6]:


## Exercice 3

window_shape = (3,3)
B = view_as_windows(v_pad,window_shape)
v_average = np.asarray([ [np.mean(B[i,j]) for j in range(len(B[0]))] for i in range(len(B))])


# In[7]:


print("v_ave: ",v_average)
print("v_ave size: ",v_average.shape)

# v = x + w + z
# The bigger is the window size the best is the estimation of host image because the gaussian noise has a 0 mean so if we take a big sample the mean value will be 0. So when we take a bigger windows size this noise will be removed
# Same reasoning with w which is -1 or 1 with uniform distribution so we obtain a 0 mean value.


# In[8]:


w_blind = v - v_average
linear_corr_blind = linearCorrelation(w,w_blind,density,tabu)


# In[14]:


# Put the image in double format and changing range to display them correctly with cv2
# Doing it after the linear correlation to minimize loss of information by the rounding operation.

x = cv2.normalize(x.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
y = cv2.normalize(y.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
v = cv2.normalize(v.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# Show orinal image, waterwarked image and attacked watermarked image

numpy_horizontal = np.hstack((x,np.hstack((y, v))))

window2 = 'Orignal vs Watermarked image vs Attacked image'
cv2.imshow(window2,numpy_horizontal) # Show image
cv2.waitKey(0)


# In[11]:


print(linear_corr)
print(linear_corr_blind)

