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



