import math
import numpy as np
import copy
import h5py
import torch

def load_test_data(idx, filetest=None, dataset="test_dataset", device='cpu'):

    img = filetest[dataset][idx]
    img = np.array(img).astype(np.float32)
    img = img[None, :, :]
    img = (torch.from_numpy(img).to(device)).permute(0, 3, 1, 2)

    return img

def load_train_data(idx, is_testing=False, batch_size=1, fileA=None, fileB=None, 
    dataset="train_dataset", device='cpu'):

    img_A = fileA[dataset][idx*batch_size:(idx+1)*batch_size]
    img_A = np.array(img_A).astype(np.float32)

    img_B = fileB[dataset][idx*batch_size:(idx+1)*batch_size]
    img_B = np.array(img_B).astype(np.float32)

    if not is_testing:
        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    img_AB = np.concatenate((img_A, img_B), axis=3)

    img_AB = (torch.from_numpy(img_AB).to(device)).permute(0, 3, 1, 2)


    return img_AB

