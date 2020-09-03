# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: August, 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Run FWI on Sleipner data



## Module loading

using DrWatson
@quickactivate "UncertaintyQuantificationAVP"
using UncertaintyQuantificationAVP
using LinearAlgebra, PyPlot, Statistics, Optim
using Random; seed = 1; Random.seed!(seed)


## Loading model and synthetics

# Data
data_dict = wload(datadir("Sleipner", "Sleipner_data.bson"))
dat_n = data_dict[:dat_n]
σ2_n = data_dict[:σ2_n]
src_pars = data_dict[:src_pars]
abc_geom = data_dict[:abc_geom]
acc = data_dict[:acc_mod]

# Model
model_dict = wload(projectdir()*"/scripts/Sleipner/Sleipner_model.bson")
Mtrue = model_dict[:Mtrue]
Mbg = model_dict[:Mbg]


## Loss function

# Pre- and post-processing for log-posterior
nz = Mbg.geom.nz
dz = Mbg.geom.dz
mmin = 1f0/3000f0^2f0
mmax = Inf32
function proj_bounds(m::Array{Float32}, mmin, mmax)
    m_ = copy(m)
    m_[m_ .< mmin] .= mmin
    m_[m_ .> mmax] .= mmax
    return m_
end
prec(m::Array{Float32, 3}) = Model(Mbg.geom, proj_bounds(reshape(m, nz, :), mmin, mmax))
post(g::Model) = reshape(g.m, (nz, 1, :))

# Log-likelihood
F = seismod_fun(dat_n.pars, src_pars; abc_geom = abc_geom, acc = acc)
negloglike_fun(m::Model; compute_grad::Bool = false) = negLogLikelihood_gauss(dat_n, m, σ2_n, F; compute_grad = compute_grad)

# Log-prior
D = derivative_linop(Mbg.geom)
Mpr = Model(Mbg.geom, zeros(Float32, nz, 1))
Dx = D.eval(Mtrue-Mpr)
ϵ2 = 1f-3*norm(Dx.m, Inf)^2f0
# σ2_pr = 1f-2*sum(sqrt.(Dx.m.^2f0.+ϵ2))[1]
# neglogpr_fun(m::Model; compute_grad::Bool = false) = negLogPrior_L1smooth(m, σ2_pr, ϵ2; m0 = Mpr, D = D, compute_grad = compute_grad)
σ2_pr = 1f-3*norm(Dx)[1]^2f0
neglogpr_fun(m::Model; compute_grad::Bool = false) = negLogPrior_gauss(m, σ2_pr; m0 = Mpr, D = D, compute_grad = compute_grad)

# Log-posterior
neglogpost_fun(m::Model; compute_grad::Bool = false) = negLogPost(m, negloglike_fun, neglogpr_fun; compute_grad = compute_grad)

# Overloading log-posterior for tensor-type input+pre-/post-conditioning
function neglogpost_fun(m::Array{Float32, 3}; compute_grad::Bool = false)
    nb = size(m, 3)
    if compute_grad
        f, g_ = neglogpost_fun(prec(m); compute_grad = compute_grad)
        return sum(f)/nb, post(g_)
    else
        f = neglogpost_fun(prec(m); compute_grad = compute_grad)
        return sum(f)/nb
    end
end
function neglogpost_fun!(F, G::Union{Nothing, Array{Float32, 3}}, m::Array{Float32, 3})
    if G == nothing
        f = neglogpost_fun(m; compute_grad = false)
    else
        f, g = neglogpost_fun(m; compute_grad = true)
        G .= g
    end
    return f
end


## Optimization

M0 = deepcopy(Mbg)
method = LBFGS()
niter = 100
optimopt = Optim.Options(iterations = niter, store_trace = true, show_trace = true, show_every = 1)
result = optimize(Optim.only_fg!(neglogpost_fun!), reshape(M0.m, :, 1, 1), method, optimopt)
Mmap = prec(Optim.minimizer(result))
fval = Optim.f_trace(result)


## Saving results

invparams_dict = Dict(:σ2_n => σ2_n, :mmin => mmin, :mmax => mmax, :ϵ2 => ϵ2, :σ2_n => σ2_n, :σ2_pr => σ2_pr, :Mpr => Mpr, :M0 => M0)
results_dict = Dict(:Mmap => Mmap, :fval => fval)
wsave(datadir("Sleipner", "Sleipner_invpars.bson"), invparams_dict)
wsave(datadir("Sleipner", "Sleipner_MAP.bson"), results_dict)
