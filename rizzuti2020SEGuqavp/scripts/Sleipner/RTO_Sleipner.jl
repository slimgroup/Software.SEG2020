# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: August, 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Run FWI on Sleipner data with added random noise



## Module loading

using DrWatson
@quickactivate "UncertaintyQuantificationAVP"
using UncertaintyQuantificationAVP
using LinearAlgebra, PyPlot, Statistics, Optim
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


## Loss function

# Pre- and post-processing for log-posterior
nz = Mbg.geom.nz
dz = Mbg.geom.dz
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
negloglike_fun(m::Model, dat::Data; compute_grad::Bool = false) = negLogLikelihood_gauss(dat, m, σ2_n, F; compute_grad = compute_grad)

# Log-prior
D = derivative_linop(Mbg.geom)
# neglogpr_fun(m::Model, Mpr::Model; compute_grad::Bool = false) = negLogPrior_L1smooth(m, σ2_pr, ϵ2; m0 = Mpr, D = D, compute_grad = compute_grad)
neglogpr_fun(m::Model, Mnoise::Model; compute_grad::Bool = false) = negLogPrior_gauss(m, σ2_pr; b = Mnoise, m0 = Mpr, D = D, compute_grad = compute_grad)

# Log-posterior
neglogpost_fun(m::Model, dat::Data, Mnoise::Model; compute_grad::Bool = false) = negLogPost(m, (m; compute_grad=compute_grad)->negloglike_fun(m, dat; compute_grad=compute_grad), (m; compute_grad=compute_grad)->neglogpr_fun(m, Mnoise; compute_grad=compute_grad); compute_grad = compute_grad)

# Overloading log-posterior for tensor-type input+pre-/post-conditioning
function neglogpost_fun(m::Array{Float32, 3}, dat::Data, Mnoise::Model; compute_grad::Bool = false)
    nb = size(m, 3)
    if compute_grad
        f, g_ = neglogpost_fun(prec(m), dat, Mnoise; compute_grad = compute_grad)
        return sum(f)/nb, post(g_)
    else
        f = neglogpost_fun(prec(m), dat, Mnoise; compute_grad = compute_grad)
        return sum(f)/nb
    end
end
function neglogpost_fun!(F, G::Union{Nothing, Array{Float32, 3}}, m::Array{Float32, 3}, dat::Data, Mnoise::Model)
    if G == nothing
        f = neglogpost_fun(m, dat, Mnoise; compute_grad = false)
    else
        f, g = neglogpost_fun(m, dat, Mnoise; compute_grad = true)
        G .= g
    end
    return f
end


## Optimization

M0 = deepcopy(Mbg)
method = LBFGS()
niter = 100
optimopt = Optim.Options(iterations = niter, store_trace = true, show_trace = true)#, show_every = 1)

n_noise = 2^7
noise_d = sqrt(σ2_n)*randn(Float32, size(dat_n.d)..., n_noise)
noise_m = sqrt(σ2_pr)*randn(Float32, size(Mpr.m)..., n_noise)
Mrto = Array{Float32, 2}(undef, nz, n_noise)
fval = Array{Array{Float32, 1}, 1}(undef, n_noise)
time = @elapsed Threads.@threads for i = 1:n_noise
    println(i, "/", n_noise)
    neglogpost_fun_n!(F, G, m) = neglogpost_fun!(F, G, m, dat_n+noise_d[:, :, :, i], Model(Mbg.geom, noise_m[:, :, i]))
    result = optimize(Optim.only_fg!(neglogpost_fun_n!), reshape(M0.m, :, 1, 1), method, optimopt)
    Mrto[:, i] = prec(Optim.minimizer(result)).m
    fval[i] = Optim.f_trace(result)
end
println(time)


## Saving results

results_dict = Dict(:noise_d => noise_d, :noise_m => noise_m, :Mrto => Mrto, :fval => fval)
wsave(datadir("Sleipner", "Sleipner_RTO.bson"), results_dict)
