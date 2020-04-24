import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import h5py
from load_vel import parihaka_model
from generator import generator
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker
sfmt=ticker.ScalarFormatter(useMathText=True) 
sfmt.set_powerlimits((0, 0))
import matplotlib

class Sample(object):
    def __init__(self, args):

        if torch.cuda.is_available() and args.cuda:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            print(' [*] GPU is available')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.objective = args.objective
        self.build_model(args)
        self.objective = args.objective

        if not os.path.exists(os.path.join(args.vel_dir, 'data/')):
            os.makedirs(os.path.join(args.vel_dir, 'data/'))

        data_path = os.path.join(args.vel_dir, 'data/small_high-freq_parihaka_model_linear-seq-shots.hdf5')
        if not os.path.isfile(data_path):
            os.system("wget https://www.dropbox.com/s/kwk483bqbyg2faq/small_high-freq_parihaka_model_linear-seq-shots.hdf5 -O" 
                + data_path)
                       
        self.y = h5py.File(data_path, 'r')["data"]

        if self.objective == 'MLE_w' or self.objective == 'MAP_w':
            self.recoveries_path = "Gzs"
        elif self.objective == 'MLE_x':
            self.recoveries_path = "Xs"
        elif self.objective == 'weak_MLE_w' or self.objective == 'weak_MAP_w':
            self.recoveries_path = "Xs"
        else:
            self.recoveries_path = ""

        if not os.path.exists(os.path.join(args.sample_dir, args.experiment, 
            self.recoveries_path)):
            os.makedirs(os.path.join(args.sample_dir, args.experiment, self.recoveries_path))


    def build_model(self, args):

        self.m0, self.m, self.dm, self.spacing, shape, origin = parihaka_model(args.vel_dir)
        self.extent = np.array([0., self.dm.shape[2]*self.spacing[0], 
            self.dm.shape[3]*self.spacing[1], 0.])/1.0e3
        self.dm = self.dm.to(self.device)
        self.load(args, os.path.join(args.checkpoint_dir, args.experiment))

    def load(self, args, checkpoint_dir):

        log_to_load = os.path.join(checkpoint_dir, 'training-logs.pt')
        assert os.path.isfile(log_to_load)

        if args.cuda == 0:
            training_logs = torch.load(log_to_load, map_location='cpu')
        else:
            training_logs = torch.load(log_to_load)

        print(' [*] Logs loaded')
        self.loss_function_x_log = training_logs['loss_function_x_log']
        self.loss_function_w_log = training_logs['loss_function_w_log']
        self.dm_loss_x_log = training_logs['dm_loss_x_log']
        self.dm_loss_w_log = training_logs['dm_loss_w_log']

        sample_str = os.path.join(args.checkpoint_dir, args.experiment, "samples.hdf5")
        assert os.path.isfile(sample_str)

        sample_file = h5py.File(sample_str, 'r')
        self.num_samples = sample_file["num_samples"][0]
        print("Number of samples: ", self.num_samples)
        self.samples = sample_file["samples"][:self.num_samples, ...]

        _ = 2 if (self.objective == 'weak_MLE_w' or self.objective == 'weak_MAP_w') else 1
        self.samples = np.squeeze(np.transpose(self.samples.reshape([self.samples.shape[0], _, 
            self.dm.shape[2], self.dm.shape[3]]), (0, 1, 3, 2)))

        for j in range(self.samples.shape[0]):
            self.samples[j, ...] = self.model_topMute(self.samples[j, ...])
        print(' [*] Samples loaded')

    def test(self, args):

        fig = plt.figure("profile", dpi=200, figsize=(7, 4))
        plt.imshow(self.dm[0, 0, :, :].t().cpu().numpy(), vmin=-4.0/100.0, vmax=4.0/100.0, 
            aspect=1, extent=self.extent, cmap="seismic", alpha=1.0, resample=True, 
            interpolation="lanczos", filterrad=1)
        plt.colorbar(fraction=0.0190, pad=0.01, format=sfmt)
        plt.xlabel("Horizontal distance (km)")
        plt.ylabel("Depth (km)")
        plt.title("True reflectivity - "  + r"$\delta \mathbf{m}$");
        plt.savefig(os.path.join(args.sample_dir, args.experiment, "dm.png"), 
            format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)


        if self.objective == 'MLE_x':
            i = 21
            fig = plt.figure("G(z_0)", dpi=100, figsize=(7, 4))
            plt.imshow(self.samples[i], vmin=-4.0/100.0, vmax=4.0/100.0, aspect=1, \
                extent=self.extent, cmap="seismic", alpha=1.0, resample=True, 
                interpolation="lanczos", filterrad=1)
            plt.colorbar(fraction=0.0190, pad=0.01, format=sfmt)
            plt.xlabel("Horizontal distance (km)")
            plt.ylabel("Depth (km)")
            plt.title(r"$\widehat{\delta \mathbf{m}}_{MLE}$")
            plt.savefig(os.path.join(args.sample_dir, args.experiment, 
                self.recoveries_path, "mle_est.png"), format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)
        elif self.objective == 'MLE_w':
            i = 150
            fig = plt.figure("G(z_0)", dpi=100, figsize=(7, 4))
            plt.imshow(self.samples[i], vmin=-4.0/100.0, vmax=4.0/100.0, \
                aspect=1, extent=self.extent, cmap="seismic", alpha=1.0, resample=True, 
                interpolation="lanczos", filterrad=1)
            plt.colorbar(fraction=0.0190, pad=0.01, format=sfmt)
            plt.xlabel("Horizontal distance (km)")
            plt.ylabel("Depth (km)")
            plt.title(r"$g(\mathbf{z},\widehat{{\mathbf{w}}}_{MLE})$" + r"$, \quad$" + \
                r"$\lambda^2=$" + ('%.0e' % 0.0))
            plt.savefig(os.path.join(args.sample_dir, args.experiment, 
                self.recoveries_path, "mle_est.png"), format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)
        elif self.objective == 'MAP_w':
            i = 150
            fig = plt.figure("G(z_0)", dpi=100, figsize=(7, 4))
            plt.imshow(self.samples[i], vmin=-4.0/100.0, vmax=4.0/100.0, \
                aspect=1, extent=self.extent, cmap="seismic", alpha=1.0, resample=True, 
                interpolation="lanczos", filterrad=1)
            plt.colorbar(fraction=0.0190, pad=0.01, format=sfmt)
            plt.xlabel("Horizontal distance (km)")
            plt.ylabel("Depth (km)")
            plt.title(r"$g(\mathbf{z},\widehat{{\mathbf{w}}}_{deep})$" + r"$, \quad$" + \
                r"$\lambda^2=$" + \
                ('%.0e' % np.round(args.weight_decay*np.prod(self.y.shape[0])/(args.eta))))
            plt.savefig(os.path.join(args.sample_dir, args.experiment, 
                self.recoveries_path, "map_est.png"), format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)  

        elif self.objective == 'weak_MLE_w' or self.objective == 'weak_MAP_w':
            i = 21
            fig = plt.figure("G(z_0)", dpi=100, figsize=(7, 4))
            plt.imshow(self.samples[i, 0], vmin=-4.0/100.0, vmax=4.0/100.0, aspect=1, \
                extent=self.extent, cmap="seismic", alpha=1.0, resample=True, 
                interpolation="lanczos", filterrad=1)
            plt.colorbar(fraction=0.0190, pad=0.01, format=sfmt)
            plt.xlabel("Horizontal distance (km)")
            plt.ylabel("Depth (km)")
            plt.title(r"$\widehat{\delta \mathbf{m}}_{weak}$" + r"$, \quad$" + 
                r"$\gamma^2=$" + \
                ('%.0e' % args.Lambda) + r"$, \quad$" + r"$\lambda^2=$" + \
                ('%.0e' % np.round(args.weight_decay)))
            plt.savefig(os.path.join(args.sample_dir, args.experiment, 
                self.recoveries_path, "x_" + str(i*args.sample_freq) + ".png"), \
                format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)            


    def model_topMute(self, image, mute_end=10, length=1):

        if len(image.shape) == 2:
            mute_start = mute_end - length
            damp = np.zeros([image.shape[0]])
            damp[0:mute_start-1] = 0.
            damp[mute_end:] = 1.
            taper_length = mute_end - mute_start + 1
            taper = (1. + \
                np.sin((np.pi/2.0*np.array(range(0,taper_length-1)))/(taper_length - 1)))/2.
            damp[mute_start:mute_end] = taper
            for j in range(0, image.shape[1]):
                image[:,j] = image[:,j]*damp

        else:
            for k in range(2):
                mute_start = mute_end - length
                damp = np.zeros([image[k, ...].shape[0]])
                damp[0:mute_start-1] = 0.
                damp[mute_end:] = 1.
                taper_length = mute_end - mute_start + 1
                taper = (1. + \
                    np.sin((np.pi/2.0*np.array(range(0,taper_length-1)))/(taper_length - 1)))/2.
                damp[mute_start:mute_end] = taper
                for j in range(0, image[k, ...].shape[1]):
                    image[k, ...][:,j] = image[k, ...][:,j]*damp
        return image