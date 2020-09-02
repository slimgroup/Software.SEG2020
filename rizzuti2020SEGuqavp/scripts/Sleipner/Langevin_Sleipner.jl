# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: March, 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Uncertainty quantification for AVP for Sleipner via Langevin dynamics



using LinearAlgebra, Statistics, PyPlot, Images, JLD2
push!(LOAD_PATH, "./src/"); using WaveOpsAVP
push!(LOAD_PATH, "../LangevinDynamics/src/"); using LangevinDynamics
using Random; Random.seed!(1)


## Load model, synthetics, and MAP result

@load "./data/Sleipner/Sleipner_data.jld"
@load "./data/Sleipner/results/MAP_results.jld"



## Setting log-posterior loss

# Pre- and post-processing for log-posterior
nz = size(Mtrue.m, 1)
mmin = R(1/3000^2)
mmax = Inf
function proj_bounds(m::Array{R}, mmin, mmax)
    m_ = copy(m)
    m_[m_ .< mmin] .= mmin
    m_[m_ .> mmax] .= mmax
    return m_
end
Mbg_ = Mbg
# Mbg_ = Mmap
# Mbg_ = Mtrue

# Log-likelihood
mloglike_fun(m::Model; compute_grad::Bool = false) = mLogLikelihood_gauss(dat_n, m, σ2_n, F; compute_grad = compute_grad)

# Log-prior
D = derivative_linop(Mbg.geom)
# D = LinFunctionMod2Mod(m->m, m->m)
M0 = Model(Mbg.geom, zeros(R, nz, 1))
# M0 = Mbg
Dx = D.eval(Mtrue-M0)
# (gauss)
# σ2_pr = R(2e-8)^2
σ2_pr = R(1e-8)^2
mlogpr_fun(m::Model; compute_grad::Bool = false) = mLogPrior_gauss(m, σ2_pr; m0 = M0, D = D, compute_grad = compute_grad)
# # (L1-smooth)
# ϵ2 = R(0.001)*norm(Dx.m, Inf)^2
# σ2_pr = R(1e-3)*sum(sqrt.(Dx.m.^R(2).+ϵ2))[1]
# mlogpr_fun(m::Model; compute_grad::Bool = false) = mLogPrior_L1smooth(m, σ2_pr, ϵ2; m0 = M0, D = D, compute_grad = compute_grad)

# Log-posterior
function mlogpost_fun(m::Array{R, 2})
    f, g = mLogPost(Model(Mbg.geom, proj_bounds(m, mmin, mmax)), mloglike_fun, mlogpr_fun; compute_grad = true)
    return f[1], g.m
end


## Langevin sampling

# SGLD parameters
maxit = 2^10
ϵ0 = R(1e-17)
ϵ1 = ϵ0*R(1e-2)
gamma = R(0.55)
# gamma = R(0)
burnin = 100
m0 = Mmap.m

# Run SGLD
time = @elapsed fval, μ, σ2, ϵ, minv = langevinSolver(mlogpost_fun, maxit, m0; ϵ0 = ϵ0, ϵ1 = ϵ1, gamma = gamma, burnin = burnin, save = true)

# Saving result
@save "./data/Sleipner/results/UQ_langevin_results.jld" dat_n σ2_pr σ2_n fval minv time μ σ2 ϵ
