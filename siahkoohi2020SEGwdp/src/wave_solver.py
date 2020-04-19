from SeisModel import ForwardBorn, AdjointBorn
from PySource import RickerSource, Receiver
from PyModel import Model
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class wave_solver(nn.Module):
    def __init__(self, shots, shape, origin, spacing, m0, dm, noise=0.0, device='cpu', sequential=False):
        super(wave_solver, self).__init__()
        
        self.forward_born = ForwardBorn()
        self.noise = noise
        self.device = device
        self.dm = dm
        self.spacing = spacing
        epsilon = np.sqrt(noise)*np.random.randn(shots.shape[0], shots.shape[1], shots.shape[2])
        self.shots = shots + epsilon

        self.model0 = Model(shape=shape, origin=origin, spacing=spacing, vp=1/np.sqrt(m0), 
            nbpml=40)
        self.T = self.mute_top(dm, mute_end=10, length=5)
        t0 = 0.
        tn = 1500.0
        dt = self.model0.critical_dt
        nt = int(1 + (tn-t0) / dt)
        self.time_range = np.linspace(t0,tn,nt)
        self.f0 = 0.030
        self.nsrc = 205
        self.nsimsrc = 205
        self.src = RickerSource(name='src', grid=self.model0.grid, f0=self.f0, time=self.time_range, 
            npoint=self.nsimsrc)
        self.src.coordinates.data[:,0] = np.linspace(0, self.model0.domain_size[0], num=self.nsimsrc)
        self.src.coordinates.data[:,-1] = 2.0*spacing[1]
        nrec = 410
        self.rec = Receiver(name='rec', grid=self.model0.grid, npoint=nrec, ntime=nt)
        self.rec.coordinates.data[:, 0] = np.linspace(0, self.model0.domain_size[0], num=nrec)
        self.rec.coordinates.data[:, 1] = 2.0*spacing[1]

        self.seq_src_idx = 0
        self.sequential = sequential

    def forward(self, x, model, src, rec, device='cpu'):
        data = self.forward_born.apply(x, model, src, rec, self.device)
        return data

    def mute_top(self, image, mute_end=10, length=1):

        mute_start = mute_end - length
        damp = torch.zeros(image.shape, device=self.device, requires_grad=False)
        damp[:, :, :, mute_end:] = 1.
        damp[:, :, :, mute_start:mute_end] = (1. + torch.sin((np.pi/2.0*torch.arange(0, length))/(length)))/2.
        def T(image, damp=damp): return damp*image
        return T

    def create_operators(self):

        if not self.sequential:
            self.mixing_weights = np.zeros([self.nsrc], dtype=np.float32)
            self.src = RickerSource(name='src', grid=self.model0.grid, f0=self.f0, time=self.time_range, 
                npoint=self.nsimsrc)
            self.src.coordinates.data[:,0] = np.linspace(0, self.model0.domain_size[0], num=self.nsimsrc)
            self.src.coordinates.data[:,-1] = 2.0*self.spacing[1]
            for s in range(self.nsrc):
                self.mixing_weights[s] = np.random.randn()/np.sqrt(self.nsrc)
                self.src.data[:, s] *= self.mixing_weights[s]
            def f(dm=self.dm, model0=self.model0, src=self.src, rec=self.rec):
                return self.forward(self.T(dm), model0, src, rec, device=self.device)
        else:
            self.src = RickerSource(name='src', grid=self.model0.grid, f0=self.f0, time=self.time_range, 
                npoint=self.nsimsrc)
            self.src.coordinates.data[:,0] = np.linspace(0, self.model0.domain_size[0], num=self.nsimsrc)
            self.src.coordinates.data[:,-1] = 2.0*self.spacing[1]
            for s in range(self.nsrc):
                if s != self.seq_src_idx:
                    self.src.data[:, s] *= 0.0
            self.seq_src_idx += 1
            def f(dm=self.dm, model0=self.model0, src=self.src, rec=self.rec):
                return self.forward(self.T(dm), model0, src, rec, device=self.device)            

        return f

    def mix_data(self):

        if not self.sequential:
            y = np.zeros(self.shots.shape[1:], dtype=np.float32)
            for s in range(self.nsrc):
                y += self.mixing_weights[s]*self.shots[s, :, :]
        else:
            y = self.shots[self.seq_src_idx-1, :, :]

        return y
