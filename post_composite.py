#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import vgi
import vgi.ei as ei

import time

QPath = 'landscape1.json'
QSet = ei.openEllipseSetFile(QPath)
sizeI, sizeOut = vgi.imageSize(QSet['imageSize'])
nH, nW, nCh = sizeI
if nCh == 1:
    sizeOut = (sizeOut[0], sizeOut[1])

img = ei.drawEllipseSet(QSet)
vgi.showImg(img, size = sizeOut)


# In[ ]:








