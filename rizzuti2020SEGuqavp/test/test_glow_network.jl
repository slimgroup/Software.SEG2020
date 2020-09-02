# HINT network from Kruse et al. (2020)
# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: August 2020

using DrWatson
@quickactivate "UncertaintyQuantificationAVP"
using UncertaintyQuantificationAVP
using InvertibleNetworks
using LinearAlgebra, Test, Random
Random.seed!(11)

# Define network
nx = 1
ny = 1
n_in = 2^3
n_hidden = 64
batchsize = 16
L = 20

# Multi-scale and single scale network
H = NetworkGlow1D(n_in, batchsize, n_hidden, L; logdet=true)

###################################################################################################
# Invertibility

# Test layers
test_size = 10
X = randn(Float32, nx, ny, n_in, test_size)

# Forward-backward
Z, logdet = H.forward(X)
_, X_ = H.backward(0f0.*Z, Z)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-3)

# Forward-inverse
Z, logdet = H.forward(X)
X_ = H.inverse(Z)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-3)

###################################################################################################
# Gradient test

# Loss
function loss(H, X)
    Z, logdet = H.forward(X)
    f = -log_likelihood(Z) - logdet
    ΔZ = -∇log_likelihood(Z)
    ΔX = H.backward(ΔZ, Z)[1]
    return f, ΔX
end

# Gradient test w.r.t. input
H = NetworkGlow1D(n_in, batchsize, n_hidden, L; logdet=true)
X = randn(Float32, nx, ny, n_in, batchsize)
X0 = randn(Float32, nx, ny, n_in, batchsize)
dX = X - X0

f0, ΔX = loss(H, X0)
h = 0.1f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test cond. HINT net: input\n")
for j=1:maxiter
    global h
    f = loss(H, X0 + h*dX)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test w.r.t. parameters
H = NetworkGlow1D(n_in, batchsize, n_hidden, L; logdet=true)
X = randn(Float32, nx, ny, n_in, batchsize); H.forward(X)
H0 = NetworkGlow1D(n_in, batchsize, n_hidden, L; logdet=true)
X0 = randn(Float32, nx, ny, n_in, batchsize); H0.forward(X0)
θ0 = get_pars(H0)
θ = get_pars(H)
dθ = θ .- θ0

f0, _ = loss(H0, X0); Δθ = deepcopy(get_grads(H0))
h = 0.1f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test cond. Glow net: pars\n")
for j=1:maxiter
    global h
    set_pars!(H0, θ0+h*dθ)
    f = loss(H0, X0)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dθ, Δθ))
    print(err1[j], "; ", err2[j], "\n")
    h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
