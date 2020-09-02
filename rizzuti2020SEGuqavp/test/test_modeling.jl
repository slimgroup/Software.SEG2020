# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: February, 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Test forward modeling



using LinearAlgebra, PyPlot
push!(LOAD_PATH, "./src/"); using WaveOpsAVP

# Model
nz = 256
dz = R(5)
m_geom = ModelGeom(nz, dz)
nb = 8
m = R(1/2000^2)*ones(R, nz, nb)
m[102:end, :] .= R(1/2000^2)
# m[52:end, 2] .= R(1/2100^2)
# m[152:end, 3] .= R(1/2200^2)
M = Model(m_geom, m)

# Data parameters
nt = 1000
dt = R(0.001)
θ = range(R(0), stop = R(sqrt(3)/2), length = 10)
p = sin.(θ)*sqrt(max(m...))
dat_pars = DataParams(nt, dt, p)

# Source parameters
wav = cfreq_wavelet(50, nt, dt, (R(5), R(10), R(30), R(60)))
src_pars = SrcParams(wav, dt)

# ABC parameters
abc_size = (20, 40)
abc_fact = R(0.0001)
abc_geom = ABC_Geom(nz, abc_size, abc_fact)

# Modeling operator
acc_mod = 4
F = seismod_fun(dat_pars, src_pars; abc_geom = abc_geom, acc = acc_mod)

# Modeling
@time dat = F.eval(M; store_extracomp = true);

# # Plot
# figure(); imshow(dat.d[:, :, 1], aspect = "auto")
