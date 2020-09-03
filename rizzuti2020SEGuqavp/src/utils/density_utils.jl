# Density utilities
# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: August 2020

export sample_gaussmixture, negLogDensity_gaussmixture

function sample_gaussmixture(nsamples::Int64, mu::Array{Float32, 2}, sigma2::Array{Float32, 1})
    nmodes = length(sigma2)
    X = Array{Float32, 4}(undef, (1, 1, 2, nsamples))
    for i = 1:nsamples
        imode = rand(1:nmodes)
        X[1, 1, :, i] = mu[imode, :]+sqrt(sigma2[imode])*randn(Float32, 2)
    end
    return X
end

function negLogDensity_gaussmixture(X::Array{Float32, 4}, mu::Array{Float32, 2}, sigma2::Array{Float32, 1}; ε::Float32 = 1f-10)
    nmodes = length(sigma2)
    p = zeros(Float32, 1, 1, 1, size(X, 4))
    g = zeros(Float32, size(X))
    for i = 1:nmodes
        p_i = exp.(-((X[:, :, 1:1, :].-mu[i, 1]).^2f0+(X[:, :, 2:2, :].-mu[i, 2]).^2f0)/(2f0*sigma2[i]))/(nmodes*2f0*pi*sigma2[i])
        p += p_i
        g += -p_i.*(X.-reshape(mu[i, :], 1, 1, 2, 1))/sigma2[i]
    end
    return -log.(p.+ε), -g./(p.+ε)
end
