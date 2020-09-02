# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: August, 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Generate synthetic Sleipner data


## Module loading

using DrWatson
@quickactivate "UncertaintyQuantificationAVP"
using UncertaintyQuantificationAVP
using LinearAlgebra, PyPlot, Statistics
using Random; seed = 1; Random.seed!(seed)


## Loading models

# Load velocity model
model_dict = wload(datadir("Sleipner", "Sleipner_model.bson"))
Mtrue = model_dict[:Mtrue]
Mbg = model_dict[:Mbg]


## Seismic experiment settings

# Data parameters
T = 0.4f0
dt = 0.0005f0
nt = Int64(ceil(T/dt))+1
θ = range(0f0, stop = pi*32f0/180f0, length = 11)
p = sin.(θ)*sqrt(max(Mtrue.m...)); p = p[5:end]
dat_pars = DataParams(nt, dt, p)

# Source parameters
zsrc = 0f0
wav = cfreq_wavelet(Int64(round(0.04f0/dt)), nt, dt, (5f0, 30f0, 50f0, 80f0))
src_pars = SrcParams(wav, dt, zsrc)

# ABC parameters
abc_size = (50, 50)
abc_fact = 1f-4
abc_geom = ABC_Geom(Mbg.geom.nz, abc_size, abc_fact)

# Modeling operator
acc_mod = 4
F = seismod_fun(dat_pars, src_pars; abc_geom = abc_geom, acc = acc_mod)

# Generate synthetics
dat = F.eval(Mtrue)
dat_bg = F.eval(Mbg)

# Adding white noise
snr = 0f0
σ2_dat = var((dat-dat_bg).d)
σ2_n = σ2_dat*10f0^(-snr/10f0)
dat_n = dat+sqrt(σ2_n)*randn(Float32, size(dat.d))


## Saving results

data_dict = Dict(:abc_geom => abc_geom, :src_pars => src_pars, :acc_mod => acc_mod, :dat => dat, :σ2_n => σ2_n, :dat_n => dat_n)
wsave(datadir("Sleipner", "Sleipner_data.bson"), data_dict)
