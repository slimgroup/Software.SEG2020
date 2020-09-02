# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: August, 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Run MALA to sample from Gaussian mixture in 2D


## Module loading

using DrWatson
@quickactivate "UncertaintyQuantificationAVP"
using UncertaintyQuantificationAVP
using LinearAlgebra, PyPlot, Statistics
using Random; Random.seed!(1)


## Setting negative log density for Gaussian mixture

# Gaussian modes parameters
nmodes = 7
muComplex = Complex{Float32}.(exp.(1im*2f0*pi*collect(range(0f0, 1f0; length=nmodes+1))))
mu = [real.(muComplex[1:end-1]) imag.(muComplex[1:end-1])]
sigma2 = zeros(Float32, nmodes); sigma2 .= Float32(0.025)

# Sampling function
sample = function sample_gaussmixture(nsamples::Int64) # Direct sampling
    x = Array{Float32, 2}(undef, (nsamples, 2))
    for i = 1:nsamples
        imode = rand(1:nmodes)
        x[i, :] = mu[imode, :]+sqrt(sigma2[imode])*randn(Float32, 2)
    end
    return x
end
x = sample(2^10)
figure()
plot(x[:, 1], x[:, 2], "*")
title("Sample from Gaussian mixture distribution")

# Negative log density
function negLogDensity_gaussmixture(x::Array{Float32, 2})
    p = zeros(Float32, size(x, 1))
    g = zeros(Float32, size(x))
    for i = 1:nmodes
        p_i = exp.(-((x[:, 1].-mu[i, 1]).^2+(x[:, 2].-mu[i, 2]).^2)/(2*sigma2[i]))/(nmodes*2f0*pi*sigma2[i])
        p += p_i
        g += -p_i.*(x.-mu[i:i, :])/sigma2[i]
    end
    return -log.(p), -g./p
end
function neglogp(x::Array{Float32, 1})
    f, g = negLogDensity_gaussmixture(reshape(x, 1, 2))
    return f[1], reshape(g, size(x))
end
function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end


## MALA sampling

# Starting input
x0 = randn(Float32, 2)

# Parameters
nsamples = 2^10
ϵ = 0.02f0
flag_MALA = true

# Run MALA
time = @elapsed fval, xLD = langevinSampler(neglogp, nsamples, x0, ϵ; flag_MALA = flag_MALA)
figure()
plot(x[:, 1], x[:, 2], "r*")
plot(xLD[1,:], xLD[2,:], ".")
title("MALA samples")
