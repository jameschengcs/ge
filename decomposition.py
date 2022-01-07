import numpy as np
import torch
import time
import vgi
from vgi.imaging import GaussImaging
from vgi.imaging import decomposite
import json
import matplotlib.pyplot as plt

n = 200
loss = 'ssimL1'
reopt_type = 2 # 0: None, 1: batch, 2: all, 3: batch+all
reopt = 10
random_type = 0
verbose = 1
target_path = 'images/MonaLisa.jpg' 
result_path = 'images/_MonaLisa.jpg' 
json_path = 'images/_MonaLisa.json' 
target = vgi.loadImage(target_path, normalize = True, gray = False )
gi = decomposite(target, n, loss = loss,
                 reopt = reopt, reopt_type = reopt_type, random_type = random_type, 
                 verbose = verbose)
print(loss, 'Err:', gi.error())
I, json_data = gi.save(result_path, json_path)
vgi.showImage(I)
