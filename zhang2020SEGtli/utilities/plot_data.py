import matplotlib.pyplot as plt
import copy
import numpy as np
import h5py
import numpy as np
from math import floor, ceil
import os
import matplotlib.pyplot as plt
font = {'family' : 'sans-serif',
        'size'   : 8}
import matplotlib
matplotlib.rc('font', **font)

class plotting_class(object):
    def __init__(self, args):

        self.frequency = args.frequency
        self.sampling_rate = args.sampling_rate
        self.result_path = args.result_path
        self.save_path = args.save_path
        self.input_data = args.input_data
        self.sampling_scheme = args.sampling_scheme
        self.num_slices = args.num_slices

        f = h5py.File(os.path.join(self.input_data, 'InterpolatedCoil_freq' + str(self.frequency) + '_A_test_' + \
            str(self.sampling_rate) + 'SamplingRate_' + str(self.sampling_scheme)) + '.hdf5', 'r')
        self.v = f['test_dataset'][:,:,:,0].astype(np.float32)

        if self.frequency == 14.66 or self.frequency == 14.33:
            self.clip = 2
            self.savePath = str(self.frequency) + 'Hz'
        elif self.frequency == 9.66 or self.frequency == 9.33:
            self.clip = 5
            self.savePath = str(self.frequency) + 'Hz'

        if not os.path.exists(os.path.join(self.save_path, self.savePath)):
            os.makedirs(os.path.join(self.save_path, self.savePath))

        self.vdata = np.zeros([self.v.shape[0], self.v.shape[0]], dtype=np.float32)
        for i in range(self.v.shape[0]):
            self.vdata[:, i] = np.reshape(self.v[i, :, :], (self.v.shape[0],))


    def genMask(self):

        if self.sampling_scheme == 'random':

            num_points = int(np.ceil(np.sqrt(self.v.shape[0]*self.sampling_rate)))+1
            step = int(np.round(self.v.shape[1]/num_points))
            start_point = 0
            end_point = self.v.shape[1] -1
            train_idx = np.linspace(start_point, end_point, num_points)
            train_idx = np.around(train_idx).astype(np.int)
            train_idx = np.arange(start_point, end_point, step)

            mask = np.zeros([self.v.shape[1], self.v.shape[1]])
            for i in train_idx:
                mask[i, train_idx] = 1.

            train_size = len(np.where(np.reshape(mask, (self.v.shape[0])) == 1.)[0])

            for i in train_idx:
                for j in train_idx:

                    u = np.random.uniform(0.0, 1.0)

                    if u >=0./9 and u<1./9 and (i-1)>=0:
                        mask[i,j] = 0.
                        mask[i-1, j] = 1.

                    elif u >=1./9 and u<2./9 and (i+1)<self.v.shape[1]:
                        mask[i,j] = 0.
                        mask[i+1, j] = 1.

                    elif u >=2./9 and u<3./9 and (j-1)>=0:
                        mask[i,j] = 0.
                        mask[i, j-1] = 1.

                    elif u >=3./9 and u<4./9 and (i-1)>=0 and (j-1)>=0:
                        mask[i,j] = 0.
                        mask[i-1, j-1] = 1.

                    elif u >=4./9 and u<5./9 and (i+1)<self.v.shape[1] and (j-1)>=0:
                        mask[i,j] = 0.
                        mask[i+1, j-1] = 1.

                    elif u >=5./9 and u<6./9 and (j+1)<self.v.shape[2]:
                        mask[i,j] = 0.
                        mask[i, j+1] = 1.

                    elif u >=6./9 and u<7./9 and (i+1)<self.v.shape[1] and (j+1)<self.v.shape[2]:
                        mask[i,j] = 0.
                        mask[i+1, j+1] = 1.

                    elif u >=7./9 and u<8./9 and (i-1)>=0 and (j+1)<self.v.shape[2]:
                        mask[i,j] = 0.
                        mask[i-1, j+1] = 1.

            self.mask = np.reshape(mask, (self.v.shape[0]))

            self.train_idx = np.where(self.mask == 1.)[0]
            self.train_size = len(self.train_idx)

        elif self.sampling_scheme=='periodic':

            num_points = int(np.ceil(np.sqrt(self.v.shape[0]*self.sampling_rate)))+1
            step = int(np.round(self.v.shape[1]/num_points))
            # start_point = int(np.floor((self.fdata.shape[1] - (num_points-1)*step + 1)/2.0))
            # end_point = self.fdata.shape[1] - start_point + 1
            start_point = 0
            end_point = self.v.shape[1] -1
            train_idx = np.linspace(start_point, end_point, num_points)
            train_idx = np.around(train_idx).astype(np.int)
            train_idx = np.arange(start_point, end_point, step)

            mask = np.zeros([self.v.shape[1], self.v.shape[1]])
            for i in train_idx:
                mask[i, train_idx] = 1.

            self.mask = np.reshape(mask, (self.v.shape[0]))

            self.train_idx = np.where(self.mask == 1.)[0]
            self.train_size = len(self.train_idx)

    def plot_sr_result(self):

        self.genMask()

        sr_subsampled = np.zeros([self.num_slices*self.v.shape[1], self.num_slices*self.v.shape[2]], \
            dtype=np.float32)
        subsampled_idx = np.where(self.mask == 0.)[0]
        sub_sampled = copy.copy(self.v)
        sub_sampled[subsampled_idx, :, :] = 0.
        kk = 0
        dataMask = np.zeros([self.num_slices*self.v.shape[1], self.num_slices*self.v.shape[2]], \
            dtype=np.float32)
        for i in range(self.num_slices):
            for j in range(self.num_slices):
                sr_subsampled[i*self.v.shape[1]:(i+1)*self.v.shape[1], 
                    j*self.v.shape[2]:(j+1)*self.v.shape[2] ] = sub_sampled[172*(86+6)+86+7 +i+j*self.v.shape[1], :, :]

                if np.linalg.norm(sub_sampled[172*(86+6)+86+7 +i+j*self.v.shape[1], :, :]) > 1.:
                    dataMask[i*self.v.shape[1]:(i+1)*self.v.shape[1], 
                        j*self.v.shape[2]:(j+1)*self.v.shape[2] ] = 1


        self.plotData(sr_subsampled[:self.num_slices*self.v.shape[1], :self.num_slices*self.v.shape[2]], \
            title='Observed data', clip=self.clip, name='Subsampled_RxSx-RySy_', xlabel='Receiver x, Source x', \
            ylabel='Receiver y, Source y', fulldata=0)

        # #####################################################

        sr_data = np.zeros([self.num_slices*self.v.shape[1], self.num_slices*self.v.shape[2]], \
            dtype=np.float32)
        mask_square = np.reshape(self.mask, (self.v.shape[1], self.v.shape[2]))
        for i in range(self.num_slices):
            for j in range(self.num_slices):
                sr_data[i*self.v.shape[1]:(i+1)*self.v.shape[1], 
                    j*self.v.shape[2]:(j+1)*self.v.shape[2] ] = self.v[172*(86+6)+86+7+i+j*self.v.shape[1], :, :]


        self.plotData(sr_data[:self.num_slices*self.v.shape[1], :self.num_slices*self.v.shape[2]], \
            title='Fully sampled data', clip=self.clip, name='FullySampled_RxSx-RySy_', xlabel='Receiver x, Source x', \
            ylabel='Receiver y, Source y', fulldata=0)    

        # #####################################################

        file_Interp = h5py.File(os.path.join(self.result_path, 'ReciprocityGAN_freq' + str(self.frequency) + '_A_train_' + \
            str(self.sampling_rate) + 'SamplingRate_' + str(self.sampling_scheme) + '_evolving_training_set_TransferLearning_from_14.33', \
            'sample', 'mapping_result.hdf5'), 'r')
        InterpResult = file_Interp["result"][:, :, :, 0].astype(np.float32)

        self.strSNR = os.path.join(self.result_path, 'ReciprocityGAN_freq' + str(self.frequency) + '_A_train_' + \
            str(self.sampling_rate) + 'SamplingRate_' + str(self.sampling_scheme) + '_evolving_training_set', \
            'sample', 'mapping_SNR.hdf5')

        sr_result = np.zeros([self.num_slices*self.v.shape[1], self.num_slices*self.v.shape[2]], \
            dtype=np.float32)

        for i in range(self.num_slices):
            for j in range(self.num_slices):
                sr_result[i*self.v.shape[1]:(i+1)*self.v.shape[1], 
                    j*self.v.shape[2]:(j+1)*self.v.shape[2] ] = InterpResult[172*(86+6)+86+7+i+j*self.v.shape[1], :, :].astype(np.float32)

        # #####################################################

        sr_result = sr_result * (1 - dataMask) + sr_subsampled * dataMask

        self.plotData(sr_result[:self.num_slices*self.v.shape[1], :self.num_slices*self.v.shape[2]], \
            title='Reconstructed data', clip=self.clip, name='Interpolation_RxSx-RySy_', xlabel='Receiver x, Source x', \
            ylabel='Receiver y, Source y', fulldata=0)
    
        self.plotData(sr_data[:self.num_slices*self.v.shape[1], :self.num_slices*self.v.shape[2]] - \
            sr_result[:self.num_slices*self.v.shape[1], :self.num_slices*self.v.shape[2]], \
            title='Reconstruction error', clip=self.clip, name='Error_RxSx-RySy_', xlabel='Receiver x, Source x', \
            ylabel='Receiver y, Source y', fulldata=0)

        # #####################################################

    def plotData(self, data, title='data', name='plot_sr_fulldata', \
        xlabel='Receiver x, Source x', ylabel='Receiver y, Source y', clip=15, fulldata=0):
        Interpolation_str = 'nearest' 

        if name=='Error_RxSx-RySy_':
            strError = ' '# + r'$\times 2$'
            data = data
        else:
            strError = ''

        if fulldata==0:
            strCropped= '_Cropped'
        else:
            strCropped= ''

        if (name=='Interpolation_RxSx-RySy_' or name=='Interpolation_RxRy-SxSy_'):
            strSNR = ' - SNR: ' +  str(round(self.readSNR(self.strSNR), 2)) + ' dB'
            Interpolation_str = 'bessel'
        else:
            strSNR = ''

        if (name=='FullySampled_RxSx-RySy_' or name=='FullySampled_RxRy-SxSy_'):
            strFullySampled = ' - Frequency: ' + str(round(self.frequency)) + 'Hz'
            if strCropped == '_Cropped':
                Interpolation_str = 'bessel'
        else:
            strFullySampled = ''

        if (name=='Subsampled_RxSx-RySy_' or name=='Subsampled_RxRy-SxSy_'):
            strSubsampling = ' - Sampling rate: ' + str(round(self.sampling_rate*100)) \
            + '%' + ' - Frequency: ' + str(round(self.frequency)) + 'Hz'
            if strCropped == '_Cropped':
                Interpolation_str = 'bessel'
        else:
            strSubsampling = ''

        if (name=='FullySampled_RxRy-SxSy_'):
            data = np.transpose(data)
        
        plt.figure(1200) 
        im = plt.imshow(np.real(data), cmap="seismic", vmin=-clip/2, vmax=clip/2, aspect='1', \
            interpolation='bessel', resample=True)
        plt.title(title + strSNR + strSubsampling + strError + strFullySampled)
        plt.tick_params(axis='both', which='both', bottom='False', top='False', labelbottom='False', \
            right='False', left='False', labelleft='False')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if name=='Error_RxSx-RySy_':
            v = np.array(np.linspace(-self.clip, self.clip, 5, endpoint=True)).astype(np.int)
            plt.colorbar(im, ticks=v, pad=.01)  


        plt.savefig(os.path.join(self.save_path, self.savePath, name + 'Frequency-' \
            +str(round(self.frequency)) + 'Hz_SamplingRate-' \
            + str(int(self.sampling_rate*100)) + 'percent_SamplingScheme-' + \
            self.sampling_scheme + strCropped + '.png'), bbox_inches='tight', dpi=1200)

    def readSNR(self, dataset):

        file_SNR = dataset
        dataset_str = "SNR"

        file_SNR = h5py.File(file_SNR, 'r')
        real = file_SNR[dataset_str][0, 0]
        print(real)
        imag = file_SNR[dataset_str][1, 0]
        print(imag)
        return real
