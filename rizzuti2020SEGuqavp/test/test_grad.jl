# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: February, 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Test gradient



using LinearAlgebra, Statistics, PyPlot, Test
using Random; Random.seed!(1)
push!(LOAD_PATH, "./src/"); using WaveOpsAVP

# Model
nz = 201
dz = R(5)
m_geom = ModelGeom(nz, dz)
nb = 32
mtrue = R(1/2000^2)*ones(R, nz, nb)
mtrue[101-50:101+50, :] .= R(1/2200^2)
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


## Log-posterior

# Setting objective functional
σ2_dat = mean(norm(dat).^R(2))
mpr = randn(R, size(mtrue)); mpr = norm(mtrue, Inf)*mpr/norm(mpr, Inf)
Mpr = Model(m_geom, mpr)
σ2_pr = mean(norm(Mpr).^R(2))
D = derivative_linop(m_geom)
fun(m::Array{R, 2}; compute_grad::Bool = false) = mlogpost_gauss(dat, Model(m_geom, m), σ2_dat, σ2_pr, F; m0 = Mpr, D = D, compute_grad = compute_grad)

# Computing gradient
m0 = R(1/2000^2)*ones(R, nz, nb)
f0, g0 = fun(m0; compute_grad = true)

# Finite-difference approx
dm = randn(R, size(m0))
dm = norm(m0, Inf)*dm/norm(dm, Inf)
t = R(1e-4)
a = (-R(1/12)*fun(m0+2*t*dm)+R(2/3)*fun(m0+t*dm)-R(2/3)*fun(m0-t*dm)+R(1/12)*fun(m0-2*t*dm))/t

# Analytic
b = dot(g0, dm)

# Relative error
@test isapprox(a, b, rtol = R(1e-6))
