# Invertible Glow network from Kingma et. al (2018)
# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: August 2020

export NetworkGlow1D, clear_grad!, get_params

"""
    H = NetworkGlow1D(nx, ny, n_in, batchsize, n_hidden, depth; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)
 Create a HINT network for data-driven generative modeling based
 on the change of variables formula.
 *Input*:
 - `nx`, `ny`, `n_in`, `batchsize`: spatial dimensions, number of channels and batchsize of input tensors `X`
 - `n_hidden`: number of hidden units in residual blocks
 - `depth`: number network layers
 - `k1`, `k2`: kernel size for first and third residual layer (`k1`) and second layer (`k2`)
 - `p1`, `p2`: respective padding sizes for residual block layers
 - `s1`, `s2`: respective strides for residual block layers
 *Output*:
 - `H`: HINT network
 *Usage:*
 - Forward mode: `Z, logdet = H.forward(X)`
 - Inverse mode: `X = H.inverse(Z)`
 - Backward mode: `ΔX, X = H.backward(ΔZ, Z)`
 *Trainable parameters:*
 - None in `H` itself
 - Trainable parameters in activation normalizations `H.AN[i]`, and in coupling layers `H.CL[i]`, where `i` ranges from `1` to `depth`.
 See also: [`ActNorm`](@ref), [`CouplingLayerHINT`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
mutable struct NetworkGlow1D <: InvertibleNetwork
    AN::AbstractArray{ActNorm, 1}
    CL::AbstractArray{CouplingLayerGlow, 1}
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor NetworkGlow1D

# Constructor
function NetworkGlow1D(n_in::Int64, batchsize::Int64, n_hidden::Int64, depth::Int64; logdet=true)

    # AN = Array{ActNorm}(undef, depth)
    AN = Array{ActNorm}(undef, depth+1)
    CL = Array{CouplingLayerGlow}(undef, depth)

    # Create layers
    for j=1:depth
        AN[j] = ActNorm(n_in; logdet=logdet)
        CL[j] = CouplingLayerGlow(1, 1, n_in, n_hidden, batchsize; k1=1, k2=1, p1=0, p2=0, logdet=logdet)
    end
    AN[depth+1] = ActNorm(n_in; logdet=logdet)

    return NetworkGlow1D(AN, CL, logdet, false)
end

# Forward pass and compute logdet
function forward(X, H::NetworkGlow1D; logdet=nothing)
    isnothing(logdet) ? logdet = (H.logdet && ~H.is_reversed) : logdet = logdet

    depth = length(H.CL)
    logdet_ = 0f0
    for j=1:depth
        logdet ? (X, logdet_an) = H.AN[j].forward(X) : X = H.AN[j].forward(X)
        logdet ? (X, logdet_cl) = H.CL[j].forward(X) : X = H.CL[j].forward(X)
        logdet && (logdet_ += (logdet_an + logdet_cl))
    end
    logdet ? (X, logdet_an) = H.AN[depth+1].forward(X) : X = H.AN[depth+1].forward(X)
    logdet && (logdet_ += logdet_an)
    logdet ? (return X, logdet_) : (return X)
end

# Inverse pass and compute gradients
function inverse(Z, H::NetworkGlow1D; logdet=nothing)
    isnothing(logdet) ? logdet = (H.logdet && H.is_reversed) : logdet = logdet

    depth = length(H.CL)
    logdet_ = 0f0
    logdet ? (Z, logdet_an) = H.AN[depth+1].inverse(Z; logdet=true) : Z = H.AN[depth+1].inverse(Z; logdet=false)
    logdet && (logdet_ += logdet_an)
    for j=depth:-1:1
        logdet ? (Z, logdet_an) = H.CL[j].inverse(Z) : Z = H.CL[j].inverse(Z)
        logdet ? (Z, logdet_cl) = H.AN[j].inverse(Z; logdet=true) : Z = H.AN[j].inverse(Z; logdet=false)
        logdet && (logdet_ += (logdet_an + logdet_cl))
    end
    logdet ? (return Z, logdet_) : (return Z)
end

# Backward pass and compute gradients
function backward(ΔZ, Z, H::NetworkGlow1D)
    depth = length(H.CL)
    ΔZ, Z = H.AN[depth+1].backward(ΔZ, Z)
    for j=depth:-1:1
        ΔZ, Z = H.CL[j].backward(ΔZ, Z)
        ΔZ, Z = H.AN[j].backward(ΔZ, Z)
    end
    return ΔZ, Z
end

# Clear gradients
function clear_grad!(H::NetworkGlow1D)
    depth = length(H.CL)
    for j=1:depth
        clear_grad!(H.AN[j])
        clear_grad!(H.CL[j])
    end
    clear_grad!(H.AN[depth+1])
end

# Get parameters
function get_params(H::NetworkGlow1D)
    depth = length(H.CL)
    p = []
    for j=1:depth
        p = cat(p, get_params(H.AN[j]); dims=1)
        p = cat(p, get_params(H.CL[j]); dims=1)
    end
    p = cat(p, get_params(H.AN[depth+1]); dims=1)
    return p
end

# Set is_reversed flag in full network tree
function tag_as_reversed!(H::NetworkGlow1D, tag::Bool)
    depth = length(H.CL)
    H.is_reversed = tag
    for j=1:depth
        tag_as_reversed!(H.AN[j], tag)
        tag_as_reversed!(H.CL[j], tag)
    end
    tag_as_reversed!(H.AN[depth+1], tag)
    return H
end
