import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import h5py
from module import generator, discriminator
from tensorboardX import SummaryWriter
from ops import weights_init_xavier_normal
from math import floor
from random import shuffle
import matplotlib.ticker as ticker
from genDataset import genDataset
from utils import load_test_data, load_train_data
from tqdm import tqdm

class wavefield_reconstruction(object):
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

        self.data_generator = genDataset(args.frequency, args.sampling_rate, args.data_path, 
            args.data_path, args.sampling_scheme)

        self.data_generator.genMask(mask_sampling='random')

        if os.path.isfile(os.path.join(args.data_path, 'InterpolatedCoil_freq' + \
            str(args.frequency) + '_A_train.hdf5')):
            os.remove(os.path.join(args.data_path, 'InterpolatedCoil_freq' + \
                str(args.frequency) + '_A_train.hdf5'))
        if os.path.isfile(os.path.join(args.data_path, 'InterpolatedCoil_freq' + \
            str(args.frequency) + '_B_train.hdf5')):
            os.remove(os.path.join(args.data_path, 'InterpolatedCoil_freq' + \
                str(args.frequency) + '_B_train.hdf5'))
        self.data_generator.genHDF5dataset()
        self.data_generator.saveTrainDataset()

        self.file_name_trainA = os.path.join(args.data_path, 'InterpolatedCoil_freq' + 
            str(args.frequency) + '_A_train.hdf5')
        self.file_name_trainB = os.path.join(args.data_path, 'InterpolatedCoil_freq' + 
            str(args.frequency) + '_B_train.hdf5')

        self.file_name_testA  = os.path.join(args.data_path, 'InterpolatedCoil_freq' + 
            str(args.frequency) + '_A_test_0.1SamplingRate_random.hdf5')
        self.file_name_testB  = os.path.join(args.data_path, 'InterpolatedCoil_freq' + 
            str(args.frequency) + '_B_test_0.1SamplingRate_random.hdf5')
        self.file_name_mask   = os.path.join(args.data_path, 'InterpolatedCoil_freq' + 
            str(args.frequency) + '_Mask_0.1SamplingRate_random.hdf5')
        self.file_name_training_mask   = os.path.join(args.data_path, 'training_mask.hdf5')

        self.file_trainA = h5py.File(self.file_name_trainA, 'r')
        self.file_trainB = h5py.File(self.file_name_trainB, 'r')
        self.file_testA  = h5py.File(self.file_name_testA, 'r')
        self.file_testB  = h5py.File(self.file_name_testB, 'r')
        self.file_mask   = h5py.File(self.file_name_mask, 'r')
        self.file_training_mask = h5py.File(self.file_name_training_mask, 'r')

        self.train_set_size  = self.file_trainA["train_dataset"].shape[0]
        self.test_set_size  = self.file_testB["test_dataset"].shape[0]

        self.L1_lambda = args.L1_lambda

        self._build_model(args)


    def _build_model(self, args):

        self.D = discriminator()
        self.G = generator()

        self.l1_loss = torch.nn.L1Loss().to(self.device)
        self.l2_loss = torch.nn.MSELoss().to(self.device)
        self.optim_G = torch.optim.Adam(self.D.parameters(), float(args.lr))
        self.optim_D = torch.optim.Adam(self.G.parameters(), float(args.lr))

        self.load(args, os.path.join(args.checkpoint_dir, args.experiment))


    def signal_to_noise(self, x, xhat):
        SNR = -20.0* torch.log(torch.norm(x - xhat)/torch.norm(x))/torch.log(torch.Tensor([10.0]))
        return SNR.item()


    def decay_lr(self, args):
        lr = args.lr if self.current_epoch < args.epoch_step else args.lr*(args.epoch-self.current_epoch)/(args.epoch-
            args.epoch_step)
        for pg in self.optim_G.param_groups:
            pg['lr'] = lr
        for pg in self.optim_D.param_groups:
            pg['lr'] = lr


    def G_objective(self, training_batch, mask):
        full_data    = training_batch[:, :2, ...]
        partial_data = training_batch[:, 2:, ...]
        pred_data = self.G(partial_data)
        pred_prob = self.D(pred_data)
        obj = self.l2_loss(pred_prob, torch.ones_like(pred_prob)) + self.L1_lambda * \
            self.l1_loss(pred_data*mask, full_data*mask)
        return obj, pred_prob


    def D_objective(self, training_batch, pred_prob):
        full_data    = training_batch[:, :2, ...]
        true_prob = self.D(full_data)
        obj = (self.l2_loss(true_prob, torch.ones_like(true_prob)) + self.l2_loss(pred_prob, 
            torch.zeros_like(pred_prob)))/2
        return obj


    def train(self, args):

        self.writer = SummaryWriter('logs/' + args.experiment)
        start_time = time.time()
        self.current_epoch = 0
        counter = 0

        mask = torch.from_numpy(1.0 - self.file_mask["mask"][0, :, :, 0]).to(self.device)
        batch_idxs = list(range(int(floor(float(self.train_set_size) / args.batch_size))))

        for self.current_epoch in range(args.epoch):

            self.file_trainA.close()
            self.file_trainB.close()
            self.file_training_mask.close()

            print(" [*] Generating training pairs ...")
            self.data_generator.genMask(mask_sampling='random')
            self.data_generator.saveTrainDataset()

            self.file_trainA = h5py.File(self.file_name_trainA, 'r')
            self.file_trainB = h5py.File(self.file_name_trainB, 'r')
            self.file_training_mask = h5py.File(self.file_name_training_mask, 'r')
            mask = torch.from_numpy(1.0 - \
                self.file_training_mask["mask"][0, :, :, 0]).to(self.device)

            shuffle(batch_idxs)
            self.decay_lr(args)

            for idx in range(0, len(batch_idxs)):       

                training_batch = load_train_data(batch_idxs[idx], batch_size=args.batch_size, \
                    fileA=self.file_trainA, fileB=self.file_trainB, dataset="train_dataset", \
                    device=self.device)

                # Update G network and record fake outputs
                G_loss, pred_prob = self.G_objective(training_batch, mask)

                grad_G = torch.autograd.grad(G_loss, self.G.parameters(), create_graph=False, retain_graph=True)
                for param, grad in zip(self.G.parameters(), grad_G):
                    param.grad = grad
                self.optim_G.step()
                self.writer.add_scalar('G_loss', G_loss, counter + 1)

                # Update D network
                D_loss = self.D_objective(training_batch, pred_prob)

                grad_D = torch.autograd.grad(D_loss, self.D.parameters(), create_graph=False)
                for param, grad in zip(self.D.parameters(), grad_D):
                    param.grad = grad
                self.optim_D.step()
                self.writer.add_scalar('D_loss', D_loss, counter + 1)

                counter += 1

                print((("Epoch: [%d/%d] | Iteration: [%d/%d] | time: %4.2f | generator loss: %f | " + 
                    "discriminator loss: %4.8f") % (self.current_epoch+1, args.epoch, idx+1, len(batch_idxs), 
                    time.time() - start_time, G_loss, D_loss)))

                if np.mod(counter, args.sample_freq) == 0:
                    self.sample_model(args.sample_dir, counter)

                if np.mod(counter, args.save_freq) == 0 or \
                    ((self.current_epoch == args.epoch - 1) and (idx == len(batch_idxs) - 1)):
                    self.save(os.path.join(args.checkpoint_dir, args.experiment), self.current_epoch)

    def save(self, checkpoint_dir, current_epoch):

        torch.save({'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict()}, os.path.join(checkpoint_dir, 
            'checkpoint_' + str(int(current_epoch)) + '.pth'))

    def load(self, args, checkpoint_dir):

        file_to_load = os.path.join(checkpoint_dir, 'checkpoint.pth')
        if os.path.isfile(file_to_load):
            if args.cuda == 0:
                checkpoint = torch.load(file_to_load, map_location='cpu')
            else:
                checkpoint = torch.load(file_to_load)
            self.G.load_state_dict(checkpoint['G_state_dict'])
            self.D.load_state_dict(checkpoint['D_state_dict'])

            print(' [*] Pre-trained networks loaded')

            self.G.train() if args.phase == 'train' else self.G.eval()

        else:
            print(' [*] No pre-trained network found')
            weights_init_xavier_normal(self.G.parameters())
            weights_init_xavier_normal(self.D.parameters())

    def sample_model(self, sample_dir, counter):

        pair_index = int(np.random.randint(0, self.train_set_size))
        mask = torch.from_numpy(1.0 - self.file_training_mask["mask"][0, :, :, 0]).to(self.device)

        train_images = load_train_data(pair_index, is_testing=True, batch_size=1, \
            fileA=self.file_trainA, fileB=self.file_trainB, dataset="train_dataset", device=self.device)

        full_data = train_images[:, :2, ...]
        partial_data = train_images[:, 2:, ...]
        
        pred_data = self.G(partial_data)
        pred_data = partial_data + pred_data*mask

        SNR = self.signal_to_noise(full_data, pred_data)
        self.writer.add_scalar('training SNR', SNR, counter)

        print(("Recovery SNR for real part (training data): %4.4f" % (SNR)))

#################################

        pair_index = int(np.random.randint(0, self.test_set_size))
        mask = torch.from_numpy(1.0 - self.file_mask["mask"][0, :, :, 0]).to(self.device)

        partial_data = load_test_data(pair_index, filetest=self.file_testB, \
            dataset="test_dataset", device=self.device)
        full_data = load_test_data(pair_index, filetest=self.file_testA, \
            dataset="test_dataset", device=self.device)

        pred_data = self.G(partial_data)
        pred_data = partial_data + pred_data*mask

        SNR = self.signal_to_noise(full_data, pred_data)
        self.writer.add_scalar('testing SNR', SNR, counter)

        print(("Recovery SNR for real part (testing data): %4.4f" % (SNR)))


    def test(self, args):

        ######################## Needs work #########################

        mask = torch.from_numpy(1.0 - self.file_mask["mask"][0, :, :, 0]).to(self.device)

        SNR_AVG = 0

        strResult = os.path.join(args.sample_dir, 'mapping_SNR.hdf5')

        if os.path.isfile(strResult):
            os.remove(strResult)

        file_SNR = h5py.File(strResult, 'w-')
        dataset_str = "SNR"
        datasetSNR = file_SNR.create_dataset(dataset_str, (1, 1))

        strCorrection = os.path.join(args.sample_dir, 'mapping_result.hdf5')

        if os.path.isfile(strCorrection):
            os.remove(strCorrection)

        file_correction = h5py.File(strCorrection, 'w-')
        datasetCorrection_str = "result"

        datasetCorrection = file_correction.create_dataset(datasetCorrection_str, 
            (self.test_set_size, self.file_testB["test_dataset"].shape[1], 
            self.file_testB["test_dataset"].shape[1], 2))

        start_time_interp = time.time()

        for itest in tqdm(range(0, self.test_set_size)):
            pair_index = itest

            partial_data = load_test_data(pair_index, filetest=self.file_testB, dataset="test_dataset", device=self.device)
            full_data = load_test_data(pair_index, filetest=self.file_testA, \
                dataset="test_dataset", device=self.device)

            pred_data = self.G(partial_data)
            datasetCorrection[itest, :, :, :] = \
                pred_data[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy()

            pred_data = partial_data + pred_data*mask

            SNR = self.signal_to_noise(full_data, pred_data)

            SNR_AVG = SNR_AVG + SNR

        datasetSNR[0, 0] = SNR_AVG/self.test_set_size
        file_SNR.close()
        file_correction.close()
