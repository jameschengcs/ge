#!/usr/bin/env python
# coding: utf-8

# In[1]:


# batch updating
import numpy as np
import vgi
import vgi.ei as ei
import time
import math 

targetPath = 'images/landscape1.jpg'
outputPath = 'landscape1.json'
n = 100
nb = 50
batchSize = nb
gamma = 0.5
rounds = int(math.ceil(n / nb)) # m
epoches = 100 # n_\tau
shrinkRate = 0.95 # \zeta
minNorm = 0.01  # \rho
reduceRound = 150 // nb # for creating H_0, larger batch index uses smaller random range.
reduceAreaRate = 0.9     

QSet, imgOut = ei.ellipseImage(targetPath = targetPath, outputPath = outputPath, 
                             nQ = nb, rounds = rounds, epoches = epoches, batchSize = batchSize, minNorm = minNorm, 
                             gamma = gamma, shrinkRate = shrinkRate, reduceRound = reduceRound, reduceAreaRate = reduceAreaRate)

vgi.showImg(imgOut)  


# In[ ]:




