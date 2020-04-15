import numpy as np
import torch
import h5py
import os

def parihaka_model(input_data=os.path.join(os.path.expanduser("~"), "data")):
    

    spacing = (25.0, 12.5)
    strName    = 'parihaka_model_high-freq.h5'
             
    
    m0 = np.transpose(h5py.File(os.path.join(input_data, strName), 'r')['m0'][...])
    dm = np.transpose(h5py.File(os.path.join(input_data, strName), 'r')['dm'][...])
    shape = h5py.File(os.path.join(input_data, strName), 'r')['n'][...][::-1]
    origin = (0., 0.)
    m = dm + m0
    dm = torch.from_numpy(dm).unsqueeze_(0).unsqueeze_(0)
    
    origin = (0., 0.)

    return m0, m, dm, spacing, shape, origin
