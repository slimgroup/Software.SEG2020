################################################################################
#
# FWI functional/gradient computation routines (python implementation using devito)
#
################################################################################

# Module loading
import numpy.linalg as npla

from twri_utils import applyfilt
from twri_propagators import forward, gradient


# Objective functional
def objFWI_devito(model, src_coords, rcv_coords, wav, dat, Filter=None, mode="eval",
                  dt=None, space_order=8):
    """
    Evaluate FWI objective functional/gradients for current m
    """
    # Setting time sampling
    if dt is None:
        dt = model.critical_dt

    # Normalization constant
    dat_filt = applyfilt(dat, Filter)
    eta = dt * npla.norm(dat_filt.reshape(-1))**2

    # Computing residual
    dmod, u = forward(model, src_coords, rcv_coords, wav, dt=dt, space_order=space_order,
                      save=(mode == "grad"))

    Pres = applyfilt(dat - dmod, Filter)

    # ||P*r||^2
    norm_Pr2 = dt * npla.norm(Pres.reshape(-1))**2

    # Functional evaluation
    fun = norm_Pr2 / eta

    # Gradient computation
    if mode == "grad":
        gradm = gradient(model, Pres, rcv_coords, u, dt=dt,
                         space_order=space_order, w=2*dt/eta)

    # Return output
    if mode == "eval":
        return fun
    elif mode == "grad":
        return fun, gradm
