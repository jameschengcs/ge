import numpy as np
import torch
import time
import vgi
from vgi.imaging import GaussImaging
from vgi.imaging import drawJson
import json
import matplotlib.pyplot as plt

json_path = 'images/_MonaLisa.json' 
max_n = 100
scale_x = 1.0
scale_y = 1.0
batch = 10
save_batch = False

primitive = 'Gaussian'
#primitive = 'ellipse'
#primitive = 'rect'
#primitive = 'brush1'
#primitive = 'brush6'
#primitive = 'brush7'
I = drawJson(json_path, max_n = max_n, batch = batch, save_batch = save_batch, primitive = primitive)
vgi.showImage(I)
