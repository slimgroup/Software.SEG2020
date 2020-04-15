import torch
import torchvision
import torch.nn as nn
from torch.autograd import Function
from JAcoustic_codegen import forward_modeling, forward_born, adjoint_born
from devito import Function as DevFunc
from devito import TimeFunction
import numpy as np

#################################################################################################################################
# Linearized forward and adjoint Born modeling as PyTorch layers
#

class ForwardBorn(Function):

    @staticmethod
    def forward(ctx, input, model, src, rec, device):
        input = input.to('cpu')
        # Save modeling parameters for backward pass
        ctx.model = model
        ctx.src = src
        ctx.rec = rec
        ctx.device = device

        # Prepare input
        input = input.detach()
        model.dm = DevFunc(name="dm", grid=model.grid)
        model.dm.data[:] = nn.ReplicationPad2d((model.nbpml))(input).numpy()[0, 0, :, :]

        # Linearized forward modeling
        d_lin = forward_born(model, src.coordinates.data, src.data, rec.coordinates.data, 
            isic=True, space_order=16)

        return torch.from_numpy(d_lin).to(ctx.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.to('cpu')
        grad_output = grad_output.detach().numpy()

        # Adjoint linearized modeling
        # u0 = forward_modeling(ctx.model, ctx.src.coordinates.data, ctx.src.data, ctx.rec.coordinates.data, 
        #     return_devito_obj=True, save=True)[1]
        u0 = forward_modeling(ctx.model, ctx.src.coordinates.data, ctx.src.data, ctx.rec.coordinates.data, 
            save=True, space_order=16)[1]
        g = adjoint_born(ctx.model, ctx.rec.coordinates.data, grad_output.data[:], u=u0, 
            isic=True, space_order=16)

        # Remove padding
        nb = ctx.model.nbpml
        g = torch.from_numpy(g[nb:-nb, nb:-nb]).to(ctx.device)

        return g.view(1, 1, g.shape[0], g.shape[1]), None, None, None, None


class AdjointBorn(Function):

    @staticmethod
    def forward(ctx, input, model, src, rec):

        # Save modeling parameters for backward pass
        ctx.model = model
        ctx.src = src
        ctx.rec = rec

        # Adjoint born modeling
        input = input.detach()
        u0 = forward_modeling(model, src.coordinates.data, src.data, rec.coordinates.data, 
            save=True, space_order=8)[1]
        g = adjoint_born(model, rec.coordinates.data, input.numpy().data[:], u=u0, 
            isic=True, space_order=8)

        # Remove padding
        nb = model.nbpml
        g = torch.from_numpy(g[nb:-nb, nb:-nb])

        return g.view(1, 1, g.shape[0], g.shape[1])

    @staticmethod
    def backward(ctx, grad_output):

        # Prepare input
        grad_output = grad_output.detach()
        ctx.model.dm = DevFunc(name="dm", grid=ctx.model.grid)
        # ctx.model.dm.data[:] = ctx.model.pad(grad_output[0,0,:,:].numpy())
        ctx.model.dm.data[:] = nn.ReflectionPad2d((40))(grad_output).numpy()[0, 0, :, :]

        # Linearized forward modeling
        d_lin = forward_born(ctx.model, ctx.src.coordinates.data, ctx.src.data,
                             ctx.rec.coordinates.data, isic=True)

        return torch.from_numpy(d_lin), None, None, None


###########################################################################################################

def absorbing_boundaries(nx, ny):

    size_mask = 30
    xMask = torch.ones(nx, 1)
    yMask = torch.ones(1, ny)
    fac = torch.tensor(0.006)

    for j in range(size_mask):
        xMask[j,0] = torch.exp(-(fac*(size_mask - j))**2)
        xMask[-1 - size_mask + j + 1,0] = torch.exp(-(fac*(size_mask-(size_mask+1+j)))**2)
        yMask[0,j] = torch.exp(-(fac*(size_mask - j))**2)
        yMask[0,-1 - size_mask + j + 1] = torch.exp(-(fac*(size_mask-(size_mask+1+j)))**2)

    Mx = torch.ones(1, 1, nx, ny)
    My = torch.ones(1, 1, nx, ny)
    for j in range(nx):
        Mx[0,0,j,:] = yMask[0,:]
    for j in range(ny):
        My[0,0,:,j] = xMask[:,0]

    return Mx*My

def ricker_wavelet(nt, dt, f0):
    t_axis = torch.linspace(0, (nt-1)*dt, nt)
    r = (np.pi*f0*(t_axis - 1/f0))
    q = (1.0 - 2.0*r**2)*torch.exp(-r**2)
    return q


def stencil(grid_spacing, order=4):

    if order==2:

        c1 = 1/grid_spacing**2
        c2 = -2/grid_spacing**2

        fd_stencil = torch.tensor([[0.0,c1,0.0], [c1,2*c2,c1], [0.0,c1,0.0]])

    elif order==4:

        c1 = -1/12/grid_spacing**2
        c2 = 4/3/grid_spacing**2
        c3 = -5/2/grid_spacing**2

        fd_stencil = torch.tensor([[0.0,0.0,c1,0.0,0.0], [0.0,0.0,c2,0.0,0.0], [c1,c2,2*c3,c2,c1],
                                  [0.0,0.0,c2,0.0,0.0], [0.0,0.0,c1,0.0,0.0]])

    elif order==20:

        c1 = -31752/293318625600/grid_spacing**2
        c2 = 784000/293318625600/grid_spacing**2
        c3 = -9426375/293318625600/grid_spacing**2
        c4 = 73872000/293318625600/grid_spacing**2
        c5 = -427329000/293318625600/grid_spacing**2
        c6 = 1969132032/293318625600/grid_spacing**2
        c7 = -7691922000/293318625600/grid_spacing**2
        c8 = 27349056000/293318625600/grid_spacing**2
        c9 = -99994986000/293318625600/grid_spacing**2
        c10 = 533306592000/293318625600/grid_spacing**2
        c11 = -909151481810/293318625600/grid_spacing**2

        fd_stencil = torch.tensor([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c1,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c2,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c3,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c4,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c5,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c6,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c7,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c8,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c9,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c10, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, 2*c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c10, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c9,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c8,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c7,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c6,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c5,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c4,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c3,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c2,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                   [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   c1,  0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
    else:
        raise Exception("Order not implemented")

    return fd_stencil
