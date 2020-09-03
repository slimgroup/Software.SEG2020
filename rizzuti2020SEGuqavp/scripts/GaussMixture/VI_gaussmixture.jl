# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: August, 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Run VI via invertible networks to sample from Gaussian mixture in 2D


## Module loading

using DrWatson
@quickactivate "UncertaintyQuantificationAVP"
using UncertaintyQuantificationAVP
using Flux.Optimise, InvertibleNetworks
using LinearAlgebra, PyPlot, Statistics
using Random; seed = 1; Random.seed!(seed)


## Setting negative log density for Gaussian mixture

# Gaussian modes parameters
nmodes = 20
R = 1f0
muComplex = Complex{Float32}.(R*exp.(1im*2f0*pi*collect(range(0f0, 1f0; length=nmodes+5))))
mu = [real.(muComplex[1:end-5]) imag.(muComplex[1:end-5])]
sigma2 = 0.025f0*ones(Float32, nmodes)

# Sampling function
sample(n) = sample_gaussmixture(n, mu, sigma2)
Xtrue = sample(1024)

# Negative log density
ε = 1f-10
neglogden(X) = negLogDensity_gaussmixture(X, mu, sigma2; ε = ε)


## Setting loss function

function loss(Z::Array{Float32, 4})
    X, logdet = T.forward(Z)
    nlogp, ΔX = neglogden(X)
    f = sum(nlogp)/batchsize-logdet
    T.backward(ΔX/batchsize, X)
    return f
end


## Network initialization

# Generator
depth = 20
n_hidden = 32
batchsize = 2^6
net_type = NetworkHINT1D
T = net_type(2, batchsize, n_hidden, depth; permute="full")
# net_type = NetworkGlow1D
# T = net_type(2, batchsize, n_hidden, depth)
initialize_id!(T)


## Training

# Training data
trainsize = 2^11
Ztrain = randn(Float32, 1, 1, 2, trainsize)

# N of iterations
nepochs = 2^7

# Stepsize
lr = 1f-3
decay = 0.9f0
nbatches = Int64(round(trainsize/batchsize))
decay_step = nbatches*2^4
# decay_step = nbatches*Int64(round(nepochs/log(1f-2)*log(decay)))
clip = 0f0
opt = Optimiser(ExpDecay(lr, decay, decay_step, clip), ADAM(lr))

# Train model
fval = Array{Float32, 1}(undef, 0)
training!(T, fval, loss, Ztrain, nepochs, batchsize, opt; gradclip=true)


## Testing

testsize = trainsize
Ztest = randn(Float32, 1, 1, 2, testsize)
Xtest, _ = T.forward(Ztest)


## Saving results

params_exp = Dict(
    :seed => seed, #seed
    :nmodes => nmodes, :mu => mu, :sigma2 => sigma2, # Gaussian mixture pars
    :batchsize => batchsize, :trainsize => trainsize, :Ztrain => Ztrain, :nepochs => nepochs, # training pars
    :lr => lr, :decay => decay, :decay_step => decay_step, :clip => clip, # optimizer pars
    :net_type => string(net_type), :depth => depth, :n_hidden => n_hidden) # network arch
results_exp = copy(params_exp)
results_exp[:fval] = fval
results_exp[:net_params] = get_params(T)
results_exp[:Ztest] = Ztest
results_exp[:Xtest] = Xtest
results_exp[:Xtrue] = Xtrue
wsave(datadir("GaussMixture", savename("VI", params_exp, "bson")), results_exp)


# ## Plotting
#
# figure()
# plot(Xtrue[1, 1, 1, :], Xtrue[1, 1, 2, :], ".")
# plot(Xtest[1, 1, 1, :], Xtest[1, 1, 2, :], "r*")
# title("VI samples")
