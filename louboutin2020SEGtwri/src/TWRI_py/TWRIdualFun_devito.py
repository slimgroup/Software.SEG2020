################################################################################
#
# TWRIdual functional/gradient computation routines (python implementation using devito)
#
################################################################################
# Module loading
import numpy as np
import numpy.linalg as npla
from devito import Inc, Function, Operator

from IPython import embed

from twri_propagators import forward, adjoint_y, gradient
from wave_utils import wavefield, wf_as_src, grad_expr
from twri_utils import applyfilt, applyfilt_transp, compute_optalpha, weight_fun


# Objective functional
def objTWRIdual_devito(model, y, src_coords, rcv_coords, wav,
                       dat, Filter, eps, mode="eval", objfact=np.float32(1),
                       comp_alpha=True, grad_corr=False, weight_fun_pars=None,
                       dt=None, space_order=8):
    """
    Evaluate TWRI objective functional/gradients for current (m, y)
    """
    # Setting time sampling
    if dt is None:
        dt = model.critical_dt

    # Computing y in reduced mode (= residual) if not provided
    u0 = None
    y_was_None = y is None
    if y_was_None:
        u0rcv, u0 = forward(model, src_coords, rcv_coords, wav, dt=dt,
                            space_order=space_order, save=(mode == "grad") and grad_corr)
        y = applyfilt(dat-u0rcv, Filter)
        PTy = applyfilt_transp(y, Filter)
    else:
        PTy = y
    # Normalization constants
    nx = np.float32(model.vp.size)
    nt, nr = np.float32(y.shape)
    etaf = npla.norm(wav.reshape(-1)) / np.sqrt(nt * nx)
    etad = npla.norm(applyfilt(dat, Filter).reshape(-1)) / np.sqrt(nt * nr)

    # Compute wavefield vy = adjoint(F(m))*Py
    norm_vPTy2, vPTy_src, vPTy = adjoint_y(model, PTy, src_coords, rcv_coords,
                                           weight_fun_pars=weight_fun_pars, dt=dt,
                                           space_order=space_order, save=(mode == "grad"))

    # <PTy, d-F(m)*f> = <PTy, d>-<adjoint(F(m))*PTy, f>
    PTy_dot_r = (np.dot(PTy.reshape(-1), dat.reshape(-1)) -
                 np.dot(vPTy_src.reshape(-1), wav.reshape(-1)))

    # ||y||
    norm_y = npla.norm(y.reshape(-1))

    # Optimal alpha
    c1 = etaf**np.float32(2) / (np.float32(4) * etad**np.float32(2) * nx * nt)
    c2 = np.float32(1) / (etad * nr * nt)
    c3 = eps / np.sqrt(nr * nt)
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
            w /= weight_fun(weight_fun_pars, model, src_coords)**2
        Q = wf_as_src(vPTy, w=w)

        # Setup gradient wrt m
        u = wavefield(model, space_order)
        gradm = Function(name="gradm", grid=model.grid)
        g_exp = grad_expr(gradm, u, vPTy, w=alpha * c2)
        rcv, _ = forward(model, src_coords, rcv_coords, wav, dt=dt,
                         space_order=space_order, q=Q, extra_expr=g_exp, u=u)

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
            gradm_corr = gradient(model, applyfilt_transp(grady_data, Filter), rcv_coords,
                                  u0, dt=dt, space_order=space_order, w=1)
            # Reduced gradient post-processing
            gradm_data = gradm.data + gradm_corr.data

    # Return output

    if mode == "eval":
        return fun / objfact
    elif mode == "grad" and y_was_None:
        return fun / objfact, gradm_data / objfact
    elif mode == "grad" and not y_was_None:
        return fun / objfact, gradm_data / objfact, grady_data / objfact
