import numpy as np
from scipy.fftpack import fft, ifft
from sympy import sqrt

from devito import TimeFunction, Function, Inc, Dimension, Eq


def wavefield(model, space_order, save=False, nt=None, fw=True):
    name = "u" if fw else "v"
    if model.is_tti:
        u = TimeFunction(name="%s1" % name, grid=model.grid, time_order=2,
                         space_order=space_order, save=None if not save else nt)
        v = TimeFunction(name="%s2" % name, grid=model.grid, time_order=2,
                         space_order=space_order, save=None if not save else nt)
        return (u, v)
    else:
        return TimeFunction(name=name, grid=model.grid, time_order=2,
                            space_order=space_order, save=None if not save else nt)


def weighted_norm(u, weight=None):
    """
    Space-time nor of a wavefield, split into norm in time first then in space to avoid
    breaking loops
    """
    if type(u) is tuple:
        expr = u[0].grid.time_dim.spacing * (u[0]**2 + u[1]**2)
        grid = u[0].grid
    else:
        expr = u.grid.time_dim.spacing * u**2
        grid = u.grid
    # Norm in time
    norm_vy2_t = Function(name="nvy2t", grid=grid)
    n_v = [Eq(norm_vy2_t, norm_vy2_t + expr)]
    # Then norm in space
    i = Dimension(name="i", )
    norm_vy2 = Function(name="nvy2", shape=(1,), dimensions=(i, ), grid=grid)
    if weight is None:
        n_v += [Eq(norm_vy2[0], norm_vy2[0] + norm_vy2_t)]
    else:
        n_v += [Eq(norm_vy2[0], norm_vy2[0] + norm_vy2_t / weight**2)]
    return norm_vy2, n_v


# Weighting
def weight_fun(weight_fun_pars, model, src_coords):
    if weight_fun_pars is None:
        return None
    if weight_fun_pars[0] == "srcfocus":
        return weight_srcfocus(model, src_coords, delta=np.float32(weight_fun_pars[1]))
    elif weight_fun_pars[0] == "depth":
        return weight_depth(model, src_coords, delta=np.float32(weight_fun_pars[1]))


def weight_srcfocus(model, src_coords, delta=np.float32(0.01)):
    """
    w(x) = sqrt((||x-xsrc||^2+delta^2)/delta^2)
    """

    ix, iz = model.grid.dimensions
    isrc = (np.float32(model.nbl) + src_coords[0, 0] / model.spacing[0],
            np.float32(model.nbl) + src_coords[0, 1] / model.spacing[1])
    h = np.sqrt(model.spacing[0]*model.spacing[1])
    return sqrt((ix-isrc[0])**2+(iz-isrc[1])**2+(delta/h)**np.float32(2))/(delta/h)


def weight_depth(model, src_coords, delta=np.float32(0.01)):
    """
    w(x) = sqrt((||z-zsrc||^2+delta^2)/delta^2)
    """

    _, iz = model.grid.dimensions
    isrc = (np.float32(model.nbl)+src_coords[0, 0]/model.spacing[0],
            np.float32(model.nbl)+src_coords[0, 1]/model.spacing[1])
    h = np.sqrt(model.spacing[0]*model.spacing[1])
    return sqrt((iz-isrc[1])**2+(delta/h)**np.float32(2))/(delta/h)


# Data filtering
def applyfilt(dat, Filter=None):
    if Filter is None:
        return dat
    else:
        pad = max(dat.shape[0], Filter.size)
        filtered = ifft(fft(dat, n=pad, axis=0)*Filter.reshape(-1, 1), axis=0)
        return np.real(filtered[:dat.shape[0], :])


def applyfilt_transp(dat, Filter=None):
    if Filter is None:
        return dat
    else:
        pad = max(dat.shape[0], Filter.size)
        filtered = ifft(fft(dat, n=pad, axis=0)*np.conj(Filter).reshape(-1, 1), axis=0)
        return np.real(filtered[:dat.shape[0], :])


# Alpha for wri
def compute_optalpha(v1, v2, v3, comp_alpha=True):

    if comp_alpha:
        if v3 < np.abs(v2):
            a = np.sign(v2)*(np.abs(v2)-v3)/(np.float32(2)*v1)
            if np.isinf(a) or np.isnan(a):
                return np.float32(0)
            else:
                return a
        else:
            return np.float32(0)
    else:
        return np.float32(1)
