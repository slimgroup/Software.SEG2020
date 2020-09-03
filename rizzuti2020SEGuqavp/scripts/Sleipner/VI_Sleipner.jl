# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: August, 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Run VI for the Sleipner model


## Module loading

using DrWatson
@quickactivate "UncertaintyQuantificationAVP"
using UncertaintyQuantificationAVP
using Flux.Optimise, InvertibleNetworks
using LinearAlgebra, PyPlot, Statistics
using Random; seed = 1; Random.seed!(seed)


## Loading synthetic data and inversion parameters

# Data
data_dict = wload(datadir("Sleipner", "Sleipner_data.bson"))
dat_n = data_dict[:dat_n]
src_pars = data_dict[:src_pars]
abc_geom = data_dict[:abc_geom]
acc = data_dict[:acc_mod]

# Model
model_dict = wload(projectdir()*"/scripts/Sleipner/Sleipner_model.bson")
Mtrue = model_dict[:Mtrue]
Mbg = model_dict[:Mbg]

# Inversion parameters
invpars_dict = wload(datadir("Sleipner", "Sleipner_invpars.bson"))
Mpr = invpars_dict[:Mpr]
M0  = invpars_dict[:M0]
σ2_n = invpars_dict[:σ2_n]
ϵ2 = invpars_dict[:ϵ2]
σ2_pr = invpars_dict[:σ2_pr]
mmin = invpars_dict[:mmin]
mmax = invpars_dict[:mmax]

# MAP result
Mmap = wload(datadir("Sleipner", "Sleipner_MAP.bson"))[:Mmap]


## Negative log density for posterior distribution

# Pre- and post-processing for log-posterior
nz = Mbg.geom.nz
dz = Mbg.geom.dz
function proj_bounds(m::Array{Float32}, mmin, mmax)
    m_ = copy(m)
    m_[m_ .< mmin] .= mmin
    m_[m_ .> mmax] .= mmax
    return m_
end
fact_tune = 1f-8
# prec(Δm::Array{Float32, 4}) = Model(Mbg.geom, proj_bounds(Mbg.m.+fact_tune*reshape(Δm, nz, :), mmin, mmax))
prec(Δm::Array{Float32, 4}) = Model(Mbg.geom, proj_bounds(Mmap.m.+fact_tune*reshape(Δm, nz, :), mmin, mmax))
post(g::Model) = fact_tune*reshape(g.m, (1, 1, nz, :))

# Log-likelihood
F = seismod_fun(dat_n.pars, data_dict[:src_pars]; abc_geom = data_dict[:abc_geom], acc = data_dict[:acc_mod])
negloglike_fun(m::Model; compute_grad::Bool = true) = negLogLikelihood_gauss(dat_n, m, σ2_n, F; compute_grad = compute_grad)

# Log-prior
D = derivative_linop(Mbg.geom)
# neglogpr_fun(m::Model; compute_grad::Bool = true) = negLogPrior_L1smooth(m, σ2_pr, ϵ2; m0 = Mpr, D = D, compute_grad = compute_grad)
neglogpr_fun(m::Model; compute_grad::Bool = true) = negLogPrior_gauss(m, σ2_pr; m0 = Mpr, D = D, compute_grad = compute_grad)

# Log-posterior
neglogpost_fun(m::Model) = negLogPost(m, negloglike_fun, neglogpr_fun; compute_grad = true)


## Setting loss function

function loss(Z::Array{Float32, 4})
    Δm, logdet = T.forward(Z)
    m = prec(Δm)
    nlogp, ∇m = neglogpost_fun(m)
    g = post(∇m)/batchsize
    f = sum(nlogp)/batchsize-logdet
    T.backward(g, Δm)
    return f
end


## Network initialization

# Generator
depth = 5
n_hidden = Mbg.geom.nz
batchsize = 2^3
net_type = NetworkHINT1D
T = net_type(nz, batchsize, n_hidden, depth; permute="full")
# net_type = NetworkGlow1D
# T = net_type(nz, batchsize, n_hidden, depth)
initialize_id!(T)


## Training

# Training data
trainsize = 2^7
Ztrain = randn(Float32, 1, 1, nz, trainsize)

# N of iterations
nepochs = 2^6

# Optimizer
lr = 1f-3
decay = 0.9f0
nbatches = Int64(round(trainsize/batchsize))
decay_step = nbatches*2^6
clip = 0f0
opt = Optimiser(ExpDecay(lr, decay, decay_step, clip), ADAM(lr))

# Train model
fval = Array{Float32, 1}(undef, 0)
training!(T, fval, loss, Ztrain, nepochs, batchsize, opt; verbose_b=true, gradclip=true)


## Testing

testsize = trainsize
Ztest = randn(Float32, 1, 1, nz, testsize)
Mtest = prec(T.forward(Ztest)[1])


## Saving results

params_exp = Dict(
    :seed => seed, #seed
    :batchsize => batchsize, :trainsize => trainsize, :Ztrain => Ztrain, :nepochs => nepochs, # training pars
    :lr => lr, :decay => decay, :decay_step => decay_step, :clip => clip, # optimizer pars
    :net_type => string(net_type), :depth => depth, :n_hidden => n_hidden) # network arch
results_exp = copy(params_exp)
results_exp[:fval] = fval
results_exp[:net_params] = get_params(T)
results_exp[:Ztest] = Ztest
results_exp[:Mtest] = Mtest
wsave(datadir("Sleipner", savename("VI", params_exp, "bson")), results_exp)
