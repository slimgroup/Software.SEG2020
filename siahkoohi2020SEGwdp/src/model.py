import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from wave_solver import *
import h5py
from load_vel import parihaka_model
from generator import generator
from tensorboardX import SummaryWriter
import matplotlib.ticker as ticker
from tqdm import tqdm
sfmt=ticker.ScalarFormatter(useMathText=True) 
sfmt.set_powerlimits((0, 0))

class LearnedImaging(object):
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

        if not os.path.exists(os.path.join(args.vel_dir, 'data/')):
            os.makedirs(os.path.join(args.vel_dir, 'data/'))

        data_path = os.path.join(args.vel_dir, 'data/small_high-freq_parihaka_model_linear-seq-shots.hdf5')
        if not os.path.isfile(data_path):
            os.system("wget https://www.dropbox.com/s/kwk483bqbyg2faq/small_high-freq_parihaka_model_linear-seq-shots.hdf5 -O" 
                + data_path)
   
        self.y = h5py.File(data_path, 'r')["data"][...]

        lin_err_std = 0.0
        self.sigma_squared = args.eta + lin_err_std**2
        self.objective = args.objective
        self.num_exps = np.prod(self.y.shape[0])
        self.epoch = torch.Tensor([args.epoch])
        self.build_model(args)
        
    def build_model(self, args):

        m0, m, self.dm, spacing, shape, origin = parihaka_model(args.vel_dir)
        self.extent = np.array([0., self.dm.shape[2]*spacing[0], 
            self.dm.shape[3]*spacing[1], 0.])/1.0e3
        self.dm = self.dm.to(self.device)
        self.Lambda = args.Lambda*np.prod(self.y.shape)/np.prod(self.dm.shape)

        dataset_size = args.epoch//args.sample_freq
        self.sample_str = os.path.join(args.checkpoint_dir, args.experiment, "samples.hdf5")
        if os.path.isfile(self.sample_str):
            os.remove(self.sample_str)

        sample_file = h5py.File(self.sample_str, 'w-')
        _ = 2 if (self.objective == 'weak_MLE_w' or self.objective == 'weak_MAP_w') else 1
        samples_h5 = sample_file.create_dataset("samples", (dataset_size, 1, _,
            self.dm.shape[2], self.dm.shape[3]), dtype=np.float32)
        num_samples_hf = sample_file.create_dataset("num_samples", [1], dtype=int)
        num_samples_hf[0] = 0
        sample_file.close()

        self.x = torch.zeros(self.dm.shape, device=self.device, requires_grad=True)
        self.wave_solver = wave_solver(self.y, shape, origin, spacing, m0, self.dm, 
            noise=args.eta, device=self.device, sequential=False)
        self.loss_function_x_log = []
        self.loss_function_w_log = []
        self.dm_loss_w_log = []
        self.dm_loss_x_log = []

        self.z = torch.randn((1, 3, 256, 192), device=self.device, requires_grad=False)
        self.G = generator(
                    self.dm.size(),
                    num_input_channels=3, num_output_channels=1, 
                    num_channels_down = [16, 32, 64],
                    num_channels_up   = [16, 32, 64],
                    num_channels_skip = [0, 0, 64],
                    upsample_mode = 'nearest',
                    need1x1_up = True,
                    filter_size_down=5,
                    filter_size_up=5,
                    filter_skip_size = 1,
                    need_sigmoid=False,
                    need_bias=True,
                    pad='reflection',
                    act_fun='LeakyReLU').to(self.device)

        self.l2_loss = torch.nn.MSELoss().to(self.device)

        self.optim_w_MAP = torch.optim.RMSprop([{'params': self.G.parameters()}], 
            float(args.lr), weight_decay=args.weight_decay)
        
        self.optim_w_MLE = torch.optim.RMSprop([{'params': self.G.parameters()}], 
            float(args.lr), weight_decay=0.0)

        self.optim_x = torch.optim.Adagrad([self.x], lr=float(args.lr)*2.0)

    def likelihood(self, pred, obs):
        return (self.y.shape[0]/(self.sigma_squared * 2.0)) * \
                    torch.norm(pred.reshape(-1) - obs.reshape(-1))**2

    def weak_prior(self, x, Gz):
        return (self.Lambda/2.0)*torch.norm(x.reshape(-1) - Gz.reshape(-1))**2

    def closure(self):
        self.optim_x.zero_grad()
        loss_function, _ = self.objective_fun('x', Lambda=self.Lambda)
        grad = self.gradient(loss_function, 'x')[0]
        self.x.grad = grad
        return loss_function

    def objective_fun(self, param_name, Lambda=0.1):

        if self.objective == 'MLE_x':
            dm_est = self.x
            pred = self.As[self.idx](self.x)
            obj = self.likelihood(pred, self.ys[self.idx])
            return obj, dm_est

        elif self.objective == 'MLE_w' or self.objective == 'MAP_w':
            dm_est = self.G(self.z)
            pred = self.As[self.idx](dm_est)
            obj = self.likelihood(pred, self.ys[self.idx])
            return obj, dm_est

        elif self.objective == 'weak_MLE_w' or self.objective == 'weak_MAP_w':
            
            if param_name == 'x':
                dm_est = self.x                
                pred = self.As[self.idx](self.x)
                obj = self.likelihood(pred, self.ys[self.idx]) + \
                        self.weak_prior(self.x, self.G(self.z))
                return obj, dm_est

            elif param_name == 'w':
                dm_est = self.G(self.z)
                obj = self.weak_prior(self.x, self.G(self.z))
                return obj, dm_est
            
            else:
                raise AssertionError()
        
        else:
            raise AssertionError()


    def gradient(self, obj, param_name):

        if param_name == 'x':
            grad = torch.autograd.grad(obj, [self.x], create_graph=False)
            return grad

        elif param_name == 'w':
            grad = torch.autograd.grad(obj, self.G.parameters(), create_graph=False)
            return grad

        else:
            raise AssertionError()


    def update_params(self, args, gradients, parameters, param_name, loss=0.0):

        for param, grad in zip(parameters, gradients):
            param.grad = grad

        if param_name == 'x':
            pass # using self.optim_x.step(self.closure) instead.

        elif param_name == 'w':

            if self.objective == 'MLE_w' or self.objective == 'weak_MLE_w':
                self.optim_w_MLE.step()

            elif self.objective == 'MAP_w' or self.objective == 'weak_MAP_w':
                self.optim_w_MAP.step()

            else:
                raise AssertionError()

        else:
            raise AssertionError()


    def solve_inverse_problem(self, args):

        if self.objective == 'MLE_x':

            loss_function = self.optim_x.step(self.closure) 
            dm_est = self.x
            self.write_history(loss_function, self.x, 'x')
            
            return dm_est

        elif self.objective == 'MLE_w' or self.objective == 'MAP_w':

            loss_function, dm_est = self.objective_fun('w')
            grad = self.gradient(loss_function, 'w')
            self.update_params(args, grad, self.G.parameters(), 'w')
            self.write_history(loss_function, dm_est, 'w')

            return dm_est

        elif self.objective == 'weak_MLE_w' or self.objective == 'weak_MAP_w':

            loss_function = self.optim_x.step(self.closure) 
            dm_x_est = self.x
            self.write_history(loss_function, dm_x_est, 'x')

            for j in range(args.inner_iter):

                loss_function, dm_w_est = self.objective_fun('w', Lambda=self.Lambda)
                grad = self.gradient(loss_function, 'w')
                self.update_params(args, grad, self.G.parameters(), 'w')

                self.write_history(loss_function, dm_w_est, 'w')

            return torch.cat((dm_x_est, dm_w_est), dim=1)


        else:
            raise AssertionError()



    def train(self, args):

        self.writer = SummaryWriter('logs/' + args.experiment)
        self.start_time = time.time()
        self.current_epoch = torch.Tensor([0])
        self.samples = []

        self.As = []
        self.ys = []
        for ne in tqdm(range(self.num_exps)):
            self.As.append(self.wave_solver.create_operators())
            self.ys.append(self.wave_solver.mix_data())
            self.ys[-1] = torch.from_numpy(self.ys[-1])
            self.ys[-1] = self.ys[-1].to(self.device)

        while (self.current_epoch < self.epoch)[0]:

            self.idx = np.random.choice(self.num_exps, 1, replace=False)[0]

            dm_est = self.solve_inverse_problem(args)
            
            if torch.fmod(self.current_epoch, args.sample_freq)==0:
                self.samples.append(dm_est.detach().cpu().numpy())

            if torch.fmod(self.current_epoch, args.save_freq) == 0 \
                or self.current_epoch == self.epoch - 1:
                self.save(os.path.join(args.checkpoint_dir, args.experiment), self.current_epoch)
                self.test(args)
                self.samples = []                
            self.current_epoch += 1


    def write_history(self, loss_function, dm_est, param_name):

        dm_loss = self.l2_loss(dm_est.reshape(-1), self.dm.reshape(-1))

        print(("(updating %s) iteration: [%d/%d] | time: %4.8f | objective: %4.8f | model misfit: %4.8f" % \
            (param_name, self.current_epoch+1, self.epoch, time.time() - self.start_time, loss_function, dm_loss)))

        if param_name == 'x':
            self.loss_function_x_log.append(loss_function.detach())
            self.dm_loss_x_log.append(dm_loss.detach())
        elif param_name == 'w':
            self.loss_function_w_log.append(loss_function.detach())
            self.dm_loss_w_log.append(dm_loss.detach())
        else:
            raise AssertionError()

        self.writer.add_scalar('loss_function_' + param_name, loss_function, self.current_epoch)
        self.writer.add_scalar('dm_loss_' + param_name, dm_loss, self.current_epoch)

    def save(self, checkpoint_dir, current_epoch):

        torch.save({'loss_function_x_log': self.loss_function_x_log,
            'loss_function_w_log': self.loss_function_w_log,
            'dm_loss_x_log': self.dm_loss_x_log,
            'dm_loss_w_log': self.dm_loss_w_log}, os.path.join(checkpoint_dir,
            'training-logs.pt'))

        torch.save({'model_state_dict': self.G.state_dict(),
            'z': self.z}, os.path.join(checkpoint_dir,
            'checkpoint.pth'))

        if len(self.samples) > 0:

            sample_file = h5py.File(self.sample_str, 'r+')
            num_samples_hf = sample_file["num_samples"]

            sample_file["samples"][num_samples_hf[0]:(num_samples_hf[0] + len(self.samples)), ...] = \
                np.array(self.samples)
            num_samples_hf[0] += len(self.samples)

            sample_file.close()

    def test(self, args):
        if len(self.samples)>0:

            if not (self.objective == 'weak_MLE_w' or self.objective == 'weak_MAP_w'):
                fig = plt.figure("G(z_0)", dpi=300, figsize=(7, 2.5))
                plt.imshow(self.model_topMute(np.transpose(self.samples[-1].reshape((self.dm.cpu().numpy().shape[2], 
                    self.dm.cpu().numpy().shape[3])))), vmin=-3.0/90.0, vmax=3.0/90.0, aspect=1, extent=self.extent)
                plt.colorbar(fraction=0.1145, pad=0.01, format=sfmt)
                plt.savefig(os.path.join(args.sample_dir, args.experiment, "dm_est_" + 
                    str(self.current_epoch.item()) + ".png"), format="png", 
                    bbox_inches="tight", dpi=300)
                plt.close(fig)

            else:
                fig = plt.figure("G(z_0)", dpi=300, figsize=(7, 2.5))
                plt.imshow(self.model_topMute(np.transpose(
                    self.samples[-1][0, 0, ...].reshape((self.dm.cpu().numpy().shape[2], 
                    self.dm.cpu().numpy().shape[3])))), vmin=-3.0/90.0, vmax=3.0/90.0, aspect=1, extent=self.extent)
                plt.colorbar(fraction=0.1145, pad=0.01, format=sfmt)
                plt.savefig(os.path.join(args.sample_dir, args.experiment, "x_" + 
                    str(self.current_epoch.item()) + ".png"), format="png", 
                    bbox_inches="tight", dpi=300)
                plt.close(fig)                

                fig = plt.figure("G(z_0)", dpi=300, figsize=(7, 2.5))
                plt.imshow(self.model_topMute(np.transpose(
                    self.samples[-1][0, 1, ...].reshape((self.dm.cpu().numpy().shape[2], 
                    self.dm.cpu().numpy().shape[3])))), vmin=-3.0/90.0, vmax=3.0/90.0, aspect=1, extent=self.extent)
                plt.colorbar(fraction=0.1145, pad=0.01, format=sfmt)
                plt.savefig(os.path.join(args.sample_dir, args.experiment, "Gz0_" + 
                    str(self.current_epoch.item()) + ".png"), format="png", 
                    bbox_inches="tight", dpi=300)
                plt.close(fig)  


        if self.objective == 'MLE_x':

            fig = plt.figure("training logs - net", dpi=300, figsize=(7, 2.5))
            plt.semilogy(self.loss_function_x_log); plt.title(r"$\|\|y_{i}-A_{i}x\|\|_2^2$")
            plt.grid()
            plt.savefig(os.path.join(args.sample_dir, args.experiment, "training-loss_x.png"), 
                format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)

            fig = plt.figure("training logs - model", dpi=300, figsize=(7, 2.5))
            plt.semilogy(self.dm_loss_x_log); plt.title(r"$\|\| \delta {m} - x\|\|_2^2$")
            plt.grid()
            plt.savefig(os.path.join(args.sample_dir, args.experiment, "model-loss_x.png"), 
                format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)

        elif self.objective == 'MLE_w' or self.objective == 'MAP_w':

            fig = plt.figure("training logs - net", dpi=300, figsize=(7, 2.5))
            plt.semilogy(self.loss_function_w_log); plt.title(r"$\|\|y_{i}-A_{i}g(z_{i},w)\|\|_2^2$")
            plt.grid()
            plt.savefig(os.path.join(args.sample_dir, args.experiment, "training-loss_w.png"), 
                format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)

            fig = plt.figure("training logs - model", dpi=300, figsize=(7, 2.5))
            plt.semilogy(self.dm_loss_w_log); plt.title(r"$\|\| \delta {m} - g(z_{i},w)\|\|_2^2$")
            plt.grid()
            plt.savefig(os.path.join(args.sample_dir, args.experiment, "model-loss_w.png"), 
                format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)

        elif self.objective == 'weak_MLE_w' or self.objective == 'weak_MAP_w':

            fig = plt.figure("training logs - net", dpi=300, figsize=(7, 2.5))
            plt.semilogy(self.loss_function_x_log); plt.title(r"$\|\|y_{i}-A_{i}x\|\|_2^2$" + \
                r"$ + \lambda \|\|x - g(z_{i},w)\|\|_2^2$")
            plt.grid()
            plt.savefig(os.path.join(args.sample_dir, args.experiment, "training-loss_x.png"), 
                format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)

            fig = plt.figure("training logs - model", dpi=300, figsize=(7, 2.5))
            plt.semilogy(self.dm_loss_x_log); plt.title(r"$\|\| \delta {m} - x\|\|_2^2$")
            plt.grid()
            plt.savefig(os.path.join(args.sample_dir, args.experiment, "model-loss_x.png"), 
                format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)

            fig = plt.figure("training logs - net", dpi=300, figsize=(7, 2.5))
            plt.semilogy(self.loss_function_w_log); plt.title(r"$\|\|x - g(z_{i},w)\|\|_2^2$")
            plt.grid()
            plt.savefig(os.path.join(args.sample_dir, args.experiment, "training-loss_w.png"), 
                format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)

            fig = plt.figure("training logs - model", dpi=300, figsize=(7, 2.5))
            plt.semilogy(self.dm_loss_w_log); plt.title(r"$\|\| \delta {m} - g(z_{i},w)\|\|_2^2$")
            plt.grid()
            plt.savefig(os.path.join(args.sample_dir, args.experiment, "model-loss_w.png"), 
                format="png", bbox_inches="tight", dpi=300)
            plt.close(fig)

        else:
            raise AssertionError()

    def model_topMute(self, image, mute_end=10, length=1):

        mute_start = mute_end - length
        damp = np.zeros([image.shape[0]])
        damp[0:mute_start-1] = 0.
        damp[mute_end:] = 1.
        taper_length = mute_end - mute_start + 1
        taper = (1. + np.sin((np.pi/2.0*np.array(range(0,taper_length-1)))/(taper_length - 1)))/2.
        damp[mute_start:mute_end] = taper
        for j in range(0, image.shape[1]):
            image[:,j] = image[:,j]*damp
        return image
