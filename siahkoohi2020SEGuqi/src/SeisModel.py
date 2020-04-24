import torch
import torch.nn as nn
from torch.autograd import Function
from JAcoustic_codegen import forward_modeling, forward_born, adjoint_born
from devito import DevFunc
import numpy as np


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
        u0 = forward_modeling(ctx.model, ctx.src.coordinates.data, ctx.src.data, ctx.rec.coordinates.data, save=True, space_order=16)[1]
        g = adjoint_born(ctx.model, ctx.rec.coordinates.data, grad_output.data[:], u=u0, 
            isic=True, space_order=16)

        # Remove padding
        nb = ctx.model.nbpml
        g = torch.from_numpy(g[nb:-nb, nb:-nb]).to(ctx.device)

        return g.view(1, 1, g.shape[0], g.shape[1]), None, None, None, None
