from kernels import wave_kernel
from geom_utils import src_rec
from wave_utils import wavefield, grad_expr
from twri_utils import weight_fun, weighted_norm

from devito import Operator, Function, Inc, Constant


def create_op(expr, subs={}, name=""):
    return Operator(expr, subs=subs, name=name)


def name(model):
    return "tti" if model.is_tti else ""


# Forward propagation
def forward(model, src_coords, rcv_coords, wavelet, dt=None, space_order=8, save=False,
            q=0, grad=False, u=None, return_op=False):
    """
    Compute forward wavefield u = A(m)^{-1}*f and related quantities (u(xrcv))
    """
    # Setting adjoint wavefield
    u = u or wavefield(model, space_order, save=save, nt=wavelet.shape[0])

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, u, q=q)

    # Setup source and receiver
    geom_expr, _, rcv = src_rec(model, u, src_coords=src_coords,
                                rec_coords=rcv_coords, wavelet=wavelet)
    # extras expressions
    extras = []
    if grad:
        gradm = Function(name="gradm", grid=model.grid)
        v =  wavefield(model, space_order, save=True, nt=wavelet.shape[0], fw=False)
        w = Constant(name="w")
        extras = grad_expr(gradm, u, v, w=w)
    # Create operator and run
    subs = model.spacing_map
    op = create_op(pde + geom_expr + extras, subs=subs, name="forward"+name(model))
    if return_op:
        return op
    op()

    # Output
    if save:
        return rcv.data, u
    else:
        return rcv.data, None


def adjoint_y(model, y, src_coords, rcv_coords, weight_fun_pars=None,
              dt=None, space_order=8, save=False, return_op=False):
    """
    Compute adjoint wavefield v = adjoint(F(m))*y
    and related quantities (||v||_w, v(xsrc))
    """
    # Setting adjoint wavefield
    v = wavefield(model, space_order, save=save, nt=y.shape[0], fw=False)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, v, fw=False)

    # Setup source and receiver
    geom_expr, _, rcv = src_rec(model, v, src_coords=rcv_coords,
                                rec_coords=src_coords, wavelet=y, fw=False)

    # Setup ||v||_w computation
    weights = weight_fun(weight_fun_pars, model, src_coords)
    norm_v, norm_v_expr = weighted_norm(v, weight=weights)
    # Create operator and run
    subs = model.spacing_map
    op = create_op(pde + geom_expr + norm_v_expr, subs=subs, name="adjoint_y"+name(model))
    if return_op:
        return op
    op()

    # Output
    if save:
        return norm_v.data[0], rcv.data, v
    else:
        return norm_v.data[0], rcv.data, None


def gradient(model, residual, rcv_coords, u, dt=None, space_order=8, w=1, return_op=False,
             w_symb=False):
    """
    Compute adjoint wavefield v = adjoint(F(m))*y
    and related quantities (||v||_w, v(xsrc))
    """
    if w_symb:
        w = Constant(name="w", value=w)
    # Setting adjoint wavefield
    v = wavefield(model, space_order, fw=False)
    u = u or wavefield(model, space_order, save=True, nt=residual.shape[0])

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, v, fw=False)

    # Setup source and receiver
    geom_expr, _, _ = src_rec(model, v, src_coords=rcv_coords,
                              wavelet=residual, fw=False)

    # Setup gradient wrt m
    gradm = Function(name="gradm", grid=model.grid)
    g_expr = grad_expr(gradm, u, v, w=w)

    # Create operator and run
    subs = model.spacing_map
    op = create_op(pde + g_expr + geom_expr, subs=subs, name="gradient"+name(model))
    if return_op:
        return op
    op()

    # Output
    return gradm.data
