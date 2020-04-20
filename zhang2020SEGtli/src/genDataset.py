
import matplotlib.pyplot as plt
import copy
import numpy as np
import h5py
import numpy as np
from math import floor, ceil
import argparse
import os
import matplotlib.pyplot as plt
font = {'family' : 'sans-serif',
        'size'   : 6}
import matplotlib
matplotlib.rc('font', **font)

class genDataset(object):
    def __init__(self, frequency, sampling_rate, input_data, data_path, sampling_scheme):

        self.frequency = frequency
        self.sampling_rate = sampling_rate
        self.data_path = data_path
        self.input_data = input_data
        self.sampling_scheme = sampling_scheme

        print(" [*] Reading seismic dataset ...")
        f = h5py.File(os.path.join(self.input_data, 'raw_' + str(self.frequency) + 'Hz.hdf5'), 'r')
        fdata = f['result'][...].astype(np.float32)
        
        self.v = fdata.shape[:-1]

        self.vdata = np.zeros([self.v[0], self.v[0]], dtype=np.complex64)
        for i in range(self.v[0]):
            self.vdata[:, i] = np.reshape(fdata[i,:,:, 0] + fdata[i,:,:, 1]*1.0j, (self.v[0],))

        fdata = 0.0

    def genMask(self, save_mask=True, mask_sampling='random'):

        if mask_sampling == 'random':

            num_points = int(np.ceil(np.sqrt(self.v[0]*self.sampling_rate)))+1
            step = int(np.round(self.v[1]/num_points))
            start_point = 0
            end_point = self.v[1] -1
            train_idx = np.linspace(start_point, end_point, num_points)
            train_idx = np.around(train_idx).astype(np.int)
            train_idx = np.arange(start_point, end_point, step)

            mask = np.zeros([self.v[1], self.v[1]])
            for i in train_idx:
                mask[i, train_idx] = 1.

            train_size = len(np.where(np.reshape(mask, (self.v[0])) == 1.)[0])

            for i in train_idx:
                for j in train_idx:

                    u = np.random.uniform(0.0, 1.0)

                    if u >=0./9 and u<1./9 and (i-1)>=0:
                        mask[i,j] = 0.
                        mask[i-1, j] = 1.

                    elif u >=1./9 and u<2./9 and (i+1)<self.v[1]:
                        mask[i,j] = 0.
                        mask[i+1, j] = 1.

                    elif u >=2./9 and u<3./9 and (j-1)>=0:
                        mask[i,j] = 0.
                        mask[i, j-1] = 1.

                    elif u >=3./9 and u<4./9 and (i-1)>=0 and (j-1)>=0:
                        mask[i,j] = 0.
                        mask[i-1, j-1] = 1.

                    elif u >=4./9 and u<5./9 and (i+1)<self.v[1] and (j-1)>=0:
                        mask[i,j] = 0.
                        mask[i+1, j-1] = 1.

                    elif u >=5./9 and u<6./9 and (j+1)<self.v[2]:
                        mask[i,j] = 0.
                        mask[i, j+1] = 1.

                    elif u >=6./9 and u<7./9 and (i+1)<self.v[1] and (j+1)<self.v[2]:
                        mask[i,j] = 0.
                        mask[i+1, j+1] = 1.

                    elif u >=7./9 and u<8./9 and (i-1)>=0 and (j+1)<self.v[2]:
                        mask[i,j] = 0.
                        mask[i-1, j+1] = 1.

            self.mask = np.reshape(mask, (self.v[0]))

            self.train_idx = np.where(self.mask == 1.)[0]
            self.train_size = len(self.train_idx)

            if save_mask:

                strMask = os.path.join(self.data_path, 'training_mask.hdf5')
                dataset_name= "mask"
                if os.path.isfile(strMask):
                    os.remove(strMask)
                self.file_Mask = h5py.File(strMask, 'w-')

                dataset_random = self.file_Mask.create_dataset(dataset_name, (1, self.v[1], self.v[2], 1))
                dataset_random[0, :, :, 0] = np.reshape(self.mask, (self.v[1], self.v[2]))


        elif mask_sampling=='periodic':

            num_points = int(np.ceil(np.sqrt(self.v[0]*self.sampling_rate)))+1
            step = int(np.round(self.v[1]/num_points))
            start_point = 0
            end_point = self.v[1] -1
            train_idx = np.linspace(start_point, end_point, num_points)
            train_idx = np.around(train_idx).astype(np.int)
            train_idx = np.arange(start_point, end_point, step)

            mask = np.zeros([self.v[1], self.v[1]])
            for i in train_idx:
                mask[i, train_idx] = 1.

            self.mask = np.reshape(mask, (self.v[0]))

            self.train_idx = np.where(self.mask == 1.)[0]
            self.train_size = len(self.train_idx)

            if save_mask:

                strMask = os.path.join(self.data_path, 'training_mask.hdf5')
                dataset_name= "mask"
                if os.path.isfile(strMask):
                    os.remove(strMask)
                self.file_Mask = h5py.File(strMask, 'w-')

                dataset_random = self.file_Mask.create_dataset(dataset_name, (1, self.v[1], self.v[2], 1))
                dataset_random[0, :, :, 0] = np.reshape(self.mask, (self.v[1], self.v[2]))

    def genHDF5dataset(self):

        strTrainA = os.path.join(self.data_path, 
            'InterpolatedCoil_freq' + str(self.frequency) + '_A_train' + '.hdf5')
        strTrainB_random = os.path.join(self.data_path, 
            'InterpolatedCoil_freq' + str(self.frequency) + '_B_train' + '.hdf5')

        dataset_train = "train_dataset"

        if not os.path.exists(strTrainA):
            self.file_TrainA = h5py.File(strTrainA, 'w-')
            self.dataset_TrainA = self.file_TrainA.create_dataset(dataset_train, (self.train_size, self.v[1], self.v[2], 2))
        else:
            self.file_TrainA = h5py.File(strTrainA, 'r+')
            self.dataset_TrainA = self.file_TrainA[dataset_train]

        if not os.path.exists(strTrainB_random):
            self.file_TrainB  = h5py.File(strTrainB_random , 'w-')
            self.dataset_TrainB = self.file_TrainB.create_dataset(dataset_train, (self.train_size, self.v[1], self.v[2], 2))
        else:
            self.file_TrainB = h5py.File(strTrainB_random, 'r+')
            self.dataset_TrainB = self.file_TrainB[dataset_train]        
        

    def saveTrainDataset(self):

        strTrainA = os.path.join(self.data_path, 
            'InterpolatedCoil_freq' + str(self.frequency) + '_A_train' + '.hdf5')
        strTrainB_random = os.path.join(self.data_path, 
            'InterpolatedCoil_freq' + str(self.frequency) + '_B_train' + '.hdf5')

        dataset_train = "train_dataset"

        self.file_TrainA = h5py.File(strTrainA, 'r+')
        self.file_TrainB  = h5py.File(strTrainB_random , 'r+')

        self.dataset_TrainA = self.file_TrainA[dataset_train]
        self.dataset_TrainB = self.file_TrainB[dataset_train]

        mask_square = np.reshape(self.mask, (self.v[1], self.v[2]))

        self.dataset_TrainA[:, :, :, 0] = np.real(np.reshape(self.vdata[self.train_idx, :], 
            (self.train_size, self.v[1], self.v[2])))
        self.dataset_TrainA[:, :, :, 1] = np.imag(np.reshape(self.vdata[self.train_idx, :], 
            (self.train_size, self.v[1], self.v[2])))
        self.file_TrainA.close()

        self.dataset_TrainB[:, :, :, 0] = np.real(np.reshape(self.vdata[self.train_idx, :], 
            (self.train_size, self.v[1], self.v[2])) * mask_square)
        self.dataset_TrainB[:, :, :, 1] = np.imag(np.reshape(self.vdata[self.train_idx, :], 
            (self.train_size, self.v[1], self.v[2])) * mask_square)
        self.file_TrainB.close()


