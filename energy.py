import numpy as np
import cv2
import copy
from utils import convolve2D

def L1(input_image):

    image = copy.deepcopy(input_image)

    xderv = np.array([-1,0,1]).reshape(1,3)
    yderv = np.array([-1,0,1]).reshape(3,1)

    dx = convolve2D(image,xderv,wpadding=1)
    dy = convolve2D(image,yderv,hpadding=1)

    out = np.abs(dx) + np.abs(dy)

    print(out.shape)

    return out

def L2(input_image):

    image = copy.deepcopy(input_image)

    xderv = np.array([-1,0,1]).reshape(1,3)
    yderv = np.array([-1,0,1]).reshape(3,1)

    dx = convolve2D(image,xderv,wpadding=1)
    dy = convolve2D(image,yderv,hpadding=1)

    out = np.sqrt(dx**2 + dy**2)

    return out

def Entropy(input_image):

    N = 5 ### considers 2Nx2N sqaure to calculate entropy of current pixel

    image = copy.deepcopy(input_image)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)       

    out = np.empty((image.shape))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            li = max(i-N,0)
            hi = min(i+N,image.shape[0])
            lj = max(j-N,0)
            hj = min(j+N,image.shape[1])

            x = image[li:hi,lj:hj].flatten()

            x = np.asarray(list(set(x)))

            n = x.shape[0]
            p = np.asarray([ np.size(x[x == i])/(1.0*n) for i in x ])

            out[i,j] = np.sum(p*np.log2(1.0/p))

    return out

def HoG(input_image):
    
    out = L1(input_image)

    hog = cv2.HOGDescriptor()
 
    h = hog.compute(input_image)

    out = out/np.max(h)

    return out

def forward_energy(input_image):
    
    image = copy.deepcopy(input_image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    M = np.zeros(image.shape)
    out = np.zeros(image.shape)
    
    Cu = np.abs( np.roll(image,-1,axis=1) - np.roll(image,1,axis=1) )
    Cl = Cu + np.abs( np.roll(image,1,axis=0) - np.roll(image,1,axis=1) )
    Cr = Cu + np.abs( np.roll(image,1,axis=0) - np.roll(image,-1,axis=1) )
    
    for i in range(1,image.shape[0]):
        
        c = np.array( [Cl[i],Cu[i],Cr[i]] )
        
        m = np.array([ np.roll(M[i-1],1) , M[i-1] , np.roll(M[i-1],-1) ]) + c
        
        ind = np.argmin(m,axis=0) ### j or j-1 or j+1
        
        M[i] = np.choose(ind,m)
        
        out[i] = np.choose(ind,c)
        
    return out