import numpy as np
# Module loading
import numpy.linalg as npla

from devito.tools import memoized_meth, as_tuple
from devito import Function, Constant, Dimension

from sources import PointSource, Receiver
from geom_utils import AcquisitionGeometry
from twri_utils import applyfilt, compute_optalpha, applyfilt_transp, weight_fun
from twri_propagators import forward ,gradient, adjoint_y
from wave_utils import wavefield, wf_as_src, grad_expr


class WRIInterface(object):

    def __init__(self, model, space_order):
        self.model = model
        self.space_order = space_order
        dummy_pos = np.zeros((1, self.model.dim))
        self.dummy_geom = AcquisitionGeometry(self.model, dummy_pos, dummy_pos, 0, 10, src_type='Ricker', f0=0.001)

    # @memoized_meth
    def op_fwd(self, save=False, q=0, grad=False):
        return forward(self.model, self.dummy_geom.src_positions, self.dummy_geom.rec_positions, self.dummy_geom.src.data,
                       space_order=self.space_order, save=save, return_op=True, q=q, grad=grad)

    # @memoized_meth
    def op_grad(self):
        return gradient(self.model, self.dummy_geom.rec.data, self.dummy_geom.rec_positions, None,
                        space_order=self.space_order, return_op=True, w_symb=True)

    # @memoized_meth
    def op_adj_y(self, weight_fun_pars=None, save=False):
        return adjoint_y(self.model, self.dummy_geom.rec.data, self.dummy_geom.src_positions, self.dummy_geom.rec_positions,
                         weight_fun_pars=weight_fun_pars, dt=None, space_order=self.space_order, save=save,
                         return_op=True)

    def forward_run(self, wav, src_coords, rcv_coords, save=False, q=0, v=None, w=0, grad=False):
        # Computing residual
        u = wavefield(self.model, self.space_order, save=save, nt=wav.shape[0])
        kwargs = wf_kwargs(u)
        rcv = Receiver(name="rcv", grid=self.model.grid, ntime=wav.shape[0], coordinates=rcv_coords)
        src = PointSource(name="src", grid=self.model.grid, ntime=wav.shape[0], coordinates=src_coords)
        src.data[:] = wav[:]
        fwd = self.op_fwd(save=save, q=q, grad=grad)
        if grad:
            w = Constant(name="w", value=w)
            gradm = Function(name="gradm", grid=self.model.grid)
            kwargs.update({as_tuple(v)[0].name: as_tuple(v)[0]})
            kwargs.update({w.name: w, gradm.name: gradm})
        fwd(rcv=rcv, src=src, **kwargs)
        if grad:
            return rcv.data, u, gradm
        return rcv.data, u

    def adjoint_y_run(self, y, src_coords, rcv_coords, weight_fun_pars=None, save=False):
         v = wavefield(self.model, self.space_order, save=save, nt=y.shape[0], fw=False)
         kwargs = wf_kwargs(v)
         rcv = Receiver(name="rcv", grid=self.model.grid, ntime=y.shape[0], coordinates=src_coords)
         src = PointSource(name="src", grid=self.model.grid, ntime=y.shape[0], coordinates=rcv_coords)
         src.data[:, :] = y[:, :]

         adj = self.op_adj_y(weight_fun_pars=weight_fun_pars, save=save)

         i = Dimension(name="i", )
         norm_v = Function(name="nvy2", shape=(1,), dimensions=(i, ), grid=self.model.grid)

         adj(src=src, rcv=rcv, nvy2=norm_v, **kwargs)

         return norm_v.data[0], rcv.data, v

    def grad_run(self, rcv_coords, Pres, u, w=1):
        kwargs = {as_tuple(u)[0].name: as_tuple(u)[0]}
        grad = self.op_grad()
        gradm = Function(name="gradm", grid=self.model.grid)
        src =  PointSource(name="src", grid=self.model.grid, ntime=Pres.shape[0], coordinates=rcv_coords)
        w = Constant(name="w", value=w)
        src.data[:, :] = Pres[:, :]
        grad(gradm=gradm, src=src, w=w, **kwargs)
        return gradm

    def fwi_fun(self, src_coords, rcv_coords, wav, dat, Filter=None, mode="eval"):
        """
        Evaluate FWI objective functional/gradients for current m
        """
        # Setting time sampling
        dt = self.model.critical_dt

        # Normalization constant
        dat_filt = applyfilt(dat, Filter)
        eta = dt * npla.norm(dat_filt.reshape(-1))**2

        # Computing residual
        dmod, u = self.forward_run(wav, src_coords, rcv_coords, save=(mode == "grad"))

        Pres = applyfilt(dat - dmod, Filter)
        # ||P*r||^2
        norm_Pr2 = dt * npla.norm(Pres.reshape(-1))**2

        # Functional evaluation
        fun = norm_Pr2 / eta

        # Gradient computation
        if mode == "grad":
            gradm = self.grad_run(rcv_coords, Pres, u, w=2*dt/eta)

        # Return output
        if mode == "eval":
            return fun
        elif mode == "grad":
            return fun, -gradm.data

    def twri_fun(self, y, src_coords, rcv_coords, wav, dat, Filter, eps, mode="eval",
                 objfact=np.float32(1), comp_alpha=True, grad_corr=False,
                 weight_fun_pars=None):
        """
        Evaluate TWRI objective functional/gradients for current (m, y)
        """
        dt = self.model.critical_dt

        # Computing y in reduced mode (= residual) if not provided
        u0 = None
        y_was_None = y is None
        if y_was_None:
            u0rcv, u0 = self.forward_run(wav, src_coords, rcv_coords,
                                         save=(mode == "grad") and grad_corr)
            y = applyfilt(dat - u0rcv, Filter)
            PTy = applyfilt_transp(y, Filter)
        else:
            PTy = y
        # Normalization constants
        nx = np.float32(self.model.vp.size)
        nt, nr = np.float32(y.shape)
        etaf = npla.norm(wav.reshape(-1)) / np.sqrt((nt * dt) * nx)
        etad = npla.norm(applyfilt(dat, Filter).reshape(-1)) / np.sqrt((nt * dt) * nr)

        # Compute wavefield vy = adjoint(F(m))*Py
        norm_vPTy2, vPTy_src, vPTy =  self.adjoint_y_run(PTy, src_coords, rcv_coords,
                                                         weight_fun_pars=weight_fun_pars,
                                                         save=(mode == "grad"))

        # <PTy, d-F(m)*f> = <PTy, d>-<adjoint(F(m))*PTy, f>
        PTy_dot_r = (np.dot(PTy.reshape(-1), dat.reshape(-1)) -
                     np.dot(vPTy_src.reshape(-1), wav.reshape(-1)))

        # ||y||
        norm_y = np.sqrt(dt) * npla.norm(y.reshape(-1))

        # Optimal alpha
        c1 = etaf**np.float32(2) / (np.float32(4) * etad**np.float32(2) * nx * (nt * dt))
        c2 = np.float32(1) / (etad * nr * (nt * dt))
        c3 = eps / np.sqrt(nr * (nt * dt))
        alpha = compute_optalpha(c1*norm_vPTy2, c2*PTy_dot_r, c3*norm_y,
                                 comp_alpha=comp_alpha)
        # Lagrangian evaluation
        fun = (alpha * (-alpha * c1 * norm_vPTy2 + c2 * PTy_dot_r) -
               np.abs(alpha) * c3 * norm_y)
        # Gradient computation
        if mode == "grad":
            # Set up extebded source
            w = 2.0 * c1 / c2 * alpha
            if weight_fun_pars is not None:
                w /= weight_fun(weight_fun_pars, self.model, src_coords)**2
            Q = wf_as_src(vPTy, w=w)

            # Setup gradient wrt m
            rcv, _, gradm = self.forward_run(wav, src_coords, rcv_coords,
                                             q=Q, grad=True, w=alpha*c2, v=vPTy)
            # Compute gradient wrt y
            if not y_was_None or grad_corr:
                norm_y = npla.norm(y)
                grady_data = alpha * c2 * applyfilt(dat - rcv.data, Filter)
                if norm_y != 0:
                    grady_data -= np.abs(alpha) * c3 * y / norm_y

            # Correcting for reduced gradient
            if not y_was_None or (y_was_None and not grad_corr):
                gradm_data = gradm.data
            else:
                gradm_corr = self.grad_run(rcv_coords, applyfilt_transp(grady_data, Filter), u0)
                # Reduced gradient post-processing
                gradm_data = gradm.data + gradm_corr.data

        # Return output
        if mode == "eval":
            return fun / objfact
        elif mode == "grad" and y_was_None:
            return fun / objfact, -gradm_data / objfact
        elif mode == "grad" and not y_was_None:
            return fun / objfact, -gradm_data / objfact, grady_data / objfact


def wf_kwargs(u):
    if type(u) is tuple:
        kwargs = {ui.name: ui for ui in u}
    else:
        kwargs = {u.name: u}
    return kwargs
