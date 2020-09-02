# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: March, 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Test seismic modeling with invertible neural network



using LinearAlgebra, Statistics, PyPlot, Test
using NNlib, Flux
import Flux.Optimise.update!
push!(LOAD_PATH, "./src/");      using WaveOpsAVP
push!(LOAD_PATH, "./src/inn1D"); using InvertibleNetworks1D
using Random; Random.seed!(1)


## Generating synthetics

# Model
nz = 256
dz = R(5)
m_geom = ModelGeom(nz, dz)
mtrue = R(1/2000^2)*ones(R, nz, 1)
mtrue[128-30:128+30, 1] .= R(1/2200^2)
Mtrue = Model(m_geom, mtrue)

# Data parameters
nt = 1001
dt = R(0.001)
θ = range(R(0), stop = R(sqrt(3)/2), length = 11)
p = sin.(θ)*sqrt(max(mtrue...))
dat_pars = DataParams(nt, dt, p)

# Source parameters
wav = cfreq_wavelet(50, nt, dt, (R(5), R(10), R(30), R(60)))
src_pars = SrcParams(wav, dt)

# ABC parameters
abc_size = (50, 50)
abc_fact = R(1e-4)
abc_geom = ABC_Geom(nz, abc_size, abc_fact)

# Modeling operator
acc_mod = 4
F = seismod_fun(dat_pars, src_pars; abc_geom = abc_geom, acc = acc_mod)

# Generate synthetics
dat = F.eval(Mtrue)


## Setting generator (multiscale invertible net architecture)

# Input size
nc = 1
batchsize = 32

# Hyperbolic net generator
depth = 2
depthH = 20
hidden_factor = 4
logdet = true
# G = HyperbolicNet1D(depth, depthH, nz, hidden_factor, batchsize; logdet = logdet);
G = HINT1D(depth, nz, hidden_factor; logdet = logdet);

# Initialization
Z0 = randn(R, nz, nc, batchsize)
G.forward(Z0)

# Tuning factor
logdet ? (Xtuning = G.forward(randn(R, nz, nc, 1000))[1]) : (Xtuning = G.forward(randn(R, nz, nc, 1000)))
normXinf = Array{R, 1}(undef, 1000)
for i = 1:1000
    normXinf[i] = norm(Xtuning[:, :, i], Inf)
end
fact_tune = R(0.1)/mean(normXinf)


## Loss function

# Log-likelihood
mloglike_fun(m::Model; compute_grad::Bool = true) = mLogLikelihood_gauss(dat_n, m, σ2_n, F; compute_grad = compute_grad)

# Log-prior
# D = derivative_linop(Mbg.geom)
D = LinFunctionMod2Mod(m->m, m->m)
# M0 = Model(Mbg.geom, zeros(R, nz, 1))
M0 = Mbg
Dx = D.eval(Mtrue-M0)
# (gauss)
σ2_pr = R(1e-8)^2
mlogpr_fun(m::Model; compute_grad::Bool = false) = mLogPrior_gauss(m, σ2_pr; m0 = M0, D = D, compute_grad = compute_grad)
# # (L1smooth)
# ϵ2 = R(0.001)*norm(Dx.m, Inf)^2
# σ2_pr = R(1e-3)*sum(sqrt.(Dx.m.^R(2).+ϵ2))[1]
# mlogpr_fun(m::Model; compute_grad::Bool = true) = mLogPrior_L1smooth(m, σ2_pr, ϵ2; m0 = M0, D = D, compute_grad = compute_grad)

# Log-posterior
mlogpost_fun(m::Model) = mLogPost(m, mloglike_fun, mlogpr_fun; compute_grad = true)


# Pre- and post-processing for log-posterior
m0 = R(1/2000^2)*ones(R, nz, 1)
prec(X::Array{R, 3}) = Model(m_geom, m0.*(R(1).+fact_tune*reshape(X, nz, batchsize))) # contrast preconditioning
post(g::Model) = fact_tune*reshape(m0.*g.m, (nz, 1, batchsize))

# Log-posterior
σ2_dat = mean(norm(dat).^R(2))
M_pr = Model(m_geom, zeros(R, size(m0)))
σ2_pr = R(1)
D = derivative_linop(m_geom)
mlogpost(m::Model) = mlogpost_gauss(dat, m, σ2_dat, σ2_pr, F; m0 = M_pr, D = D, compute_grad = true)

# Overloading log-posterior for tensor-type input
function mlogpost(X::Array{R, 3})
    f, g_ = mlogpost(prec(X))
    return f, post(g_)
end

# # Final objective
# function loss(Z::Array{R, 3})
#     nb = size(Z, 3)
#     logdet ? ((X, lgdet) = G.forward(Z)) : (X = G.forward(Z))
#     f, ΔX = mlogpost(X)
#     ΔZ, Z = G.backward(ΔX/nb, X)
#     logdet ? (return sum(f)/nb-lgdet, ΔZ) : (return sum(f)/nb, ΔZ)
# end

# Final objective (only mlogpost)
function loss(X::Array{R, 3})
    nb = size(X, 3)
    f, ΔX = mlogpost(X)
    return sum(f)/nb, ΔX/nb
end


## Gradient test (wrt input)

_, dZ0 = loss(Z0)

ΔZ = randn(R, nz, nc, batchsize)
h = R(1e-3)
a = (R(1/12)*loss(Z0-2*h*ΔZ)[1]-R(2/3)*loss(Z0-h*ΔZ)[1]+R(2/3)*loss(Z0+h*ΔZ)[1]-R(1/12)*loss(Z0+2*h*ΔZ)[1])/h
b = dot(dZ0, ΔZ)
@show err = norm(a-b)/norm(a)


# ## Parameter algebra
#
# import Base.+, Base.-, Base.*, LinearAlgebra.dot
# import InvertibleNetworks.Parameter
# function +(θ1::Array{Parameter, 1}, θ2::Array{Parameter, 1})
#     θ = deepcopy(θ1)
#     for i = 1:length(θ1)
#         θ[i].data = θ1[i].data+θ2[i].data
#         θ[i].grad = nothing
#     end
#     return θ
# end
# function -(θ1::Array{Parameter, 1}, θ2::Array{Parameter, 1})
#     θ = deepcopy(θ1)
#     for i = 1:length(θ1)
#         θ[i].data = θ1[i].data-θ2[i].data
#         θ[i].grad = nothing
#     end
#     return θ
# end
# function *(t::R, θ::Array{Parameter, 1})
#     θt = deepcopy(θ)
#     for i = 1:length(θ)
#         θt[i].data = t*θ[i].data
#         θt[i].grad = nothing
#     end
#     return θt
# end
# function dot(θ1::Array{Parameter, 1}, θ2::Array{Parameter, 1})
#     a = R(0)
#     for i = 1:length(θ1)
#         a += dot(θ1[i].data, θ2[i].data)
#     end
#     return a
# end
#
#
# ## Gradient test (wrt parameters)
#
# # Redefine loss as a function of θ
# clear_grad!(G)
# θ_ = get_params(G)
# function loss_(θ)
#     for i = 1:length(θ)
#         θ_[i].data = θ[i].data
#         θ_[i].grad = nothing
#     end
#     return loss(Z0)[1]
# end
#
# # Computing gradient
# θ0 = Array{Parameter, 1}(undef, length(θ_))
# for i = 1:length(θ0)
#     θ0[i] = Parameter(deepcopy(θ_[i].data), nothing)
# end
# f0 = loss_(θ0)
# dθ0 = Array{Parameter, 1}(undef, length(θ0))
# for i = 1:length(θ0)
#     dθ0[i] = Parameter(deepcopy(θ_[i].grad), nothing)
# end
#
# # Finite-difference approx
# Δθ = Array{Parameter, 1}(undef, length(θ0))
# for i = 1:length(θ0)
#     Δθ[i] = Parameter(nothing, nothing)
#     Δθ[i].data = randn(R, size(dθ0[i].data))
#     Δθ[i].data = Δθ[i].data*norm(θ0[i].data, Inf)/norm(Δθ[i].data, Inf)
# end
# h = R(1e-2)
# a = (-R(1/12)*loss_(θ0+2*h*Δθ)+R(2/3)*loss_(θ0+h*Δθ)-R(2/3)*loss_(θ0-h*Δθ)+R(1/12)*loss_(θ0-2*h*Δθ))/h
#
# # Analytic
# b = dot(dθ0, Δθ)
#
# # Relative error
# @show err = norm(a-b)/norm(a)
