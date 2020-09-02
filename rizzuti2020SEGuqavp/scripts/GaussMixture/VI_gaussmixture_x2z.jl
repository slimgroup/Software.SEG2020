# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: August, 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Run VI via invertible networks to sample from Gaussian mixture in 2D


## Module loading

using DrWatson
@quickactivate "UncertaintyQuantificationAVP"
using UncertaintyQuantificationAVP
using Flux, Flux.Optimise, InvertibleNetworks
using LinearAlgebra, PyPlot, Statistics
using Random; seed = 1; Random.seed!(seed)


## Setting negative log density for Gaussian mixture

# Gaussian modes parameters
nmodes = 7
muComplex = Complex{Float32}.(exp.(1im*2f0*pi*collect(range(0f0, 1f0; length=nmodes+1))))
mu = [real.(muComplex[1:end-1]) imag.(muComplex[1:end-1])]
sigma2 = 0.025f0*ones(Float32, nmodes)

# Sampling function
sample = function sample_gaussmixture(nsamples::Int64) # Direct sampling
    X = Array{Float32, 4}(undef, (1, 1, 2, nsamples))
    for i = 1:nsamples
        imode = rand(1:nmodes)
        X[1, 1, :, i] = mu[imode, :]+sqrt(sigma2[imode])*randn(Float32, 2)
    end
    return X
end
Xtrue = sample(1024)

# Negative log density
function negLogDensity_gaussmixture(X::Array{Float32, 4})
    p = zeros(Float32, 1, 1, 1, size(X, 4))
    g = zeros(Float32, size(X))
    for i = 1:nmodes
        p_i = exp.(-((X[:, :, 1:1, :].-mu[i, 1]).^2f0+(X[:, :, 2:2, :].-mu[i, 2]).^2f0)/(2f0*sigma2[i]))/(nmodes*2f0*pi*sigma2[i])
        p += p_i
        g += -p_i.*(X.-reshape(mu[i, :], 1, 1, 2, 1))/sigma2[i]
    end
    return -log.(p), -g./p
end


## Setting loss function

function loss(X::Array{Float32, 4})
    Z, logdet = T.forward(X)
    f = -log_likelihood(Z) - logdet
    ΔZ = -∇log_likelihood(Z)
    T.backward(ΔZ, Z)
    return f
end


## Network initialization

# Generator
depth = 20
n_hidden = 32
batchsize = 2^4
net_type = NetworkHINT1D
T = net_type(2, batchsize, n_hidden, depth; permute="full")
# net_type = NetworkGlow1D
# T = net_type(2, batchsize, n_hidden, depth)


## Training

# Training data
trainsize = 2^10
nbatches = Int64(round(trainsize/batchsize))
Xtrain = sample(trainsize)

# N of iterations
nepochs = 2^7

# Stepsize
lr = 1f-3
decay = 0.9f0
decay_step = nbatches*2^6
clip = 0f0
opt = Optimiser(ExpDecay(lr, decay, decay_step, clip), ADAM())
fval = Array{Float32}(undef, 0)

# Loop over iterations
time = @elapsed for n = 1:nepochs

    # Shuffling data
    idx_train_perm = randperm(trainsize)

    # Loop over batches
    for idx_batch = 1:nbatches

        idx = idx_train_perm[(idx_batch-1)*batchsize+1:idx_batch*batchsize]
        X = Xtrain[:, :, :, idx]

        # Evaluate objective and backpropagating
        l = loss(X)
        append!(fval, l)

        # Update network
        for p in get_params(T)
            update!(opt, p.data, p.grad)
        end

        # Clear gradients
        clear_grad!(T)

    end

    # Print msg
    print("Epoch: [", n, "/", nepochs, "] --- fval = ", fval[end], "\n")

end


## Testing

testsize = 1024
Ztest = randn(Float32, 1, 1, 2, testsize)
Xtest = T.inverse(Ztest)


## Saving results

params_exp = Dict(
    :seed => seed, #seed
    :nmodes => nmodes, :mu => mu, :sigma2 => sigma2, # Gaussian mixture pars
    :batchsize => batchsize, :trainsize => trainsize, :Xtrain => Xtrain, :nepochs => nepochs, # training pars
    :lr => lr, :decay => decay, :decay_step => decay_step, :clip => clip, # optimizer pars
    :net_type => string(net_type), :depth => 4, :n_hidden => n_hidden) # network arch
results_exp = copy(params_exp)
results_exp[:fval] = fval
results_exp[:net_params] = get_params(T)
results_exp[:Ztest] = Ztest
results_exp[:Xtest] = Xtest
wsave(datadir("GaussMixture", savename("VI_x2z", params_exp, "bson")), results_exp)


## Plotting

figure()
plot(Xtrue[1, 1, 1, :], Xtrue[1, 1, 2, :], ".")
plot(Xtest[1, 1, 1, :], Xtest[1, 1, 2, :], "r*")
title("VI samples")
