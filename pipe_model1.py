### This module will be used to call the h5 unet model,
### Input file will be in .mat 

import scipy.io
import numpy as np
import os 

input_path = "/Users/helen/Desktop/HL_Dev_Model/final_slices.mat"
input_data = scipy.io.loadmat('final_slices.mat')
