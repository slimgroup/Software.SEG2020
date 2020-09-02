# Invertible Hyperbolic network from Keegan et al (2018)
# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: August 2020

export NetworkHyperbolic1D, clear_grad!, get_params

"""
    H = NetworkHyperbolic1D(nx, ny, n_in, batchsize, n_hidden, depth; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)
 Create an hyperbolic network for data-driven generative modeling based
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
mutable struct NetworkHyperbolic1D <: InvertibleNetwork
    AN::AbstractArray{ActNorm, 1}
    CL::AbstractArray{HyperbolicLayer, 2}
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor NetworkHyperbolic1D

# Constructor
function NetworkHyperbolic1D(n_in::Int64, batchsize::Int64, n_hidden::Int64, depth::Int64; logdet=true, depthHL=1)

    AN = Array{ActNorm}(undef, depth)
    CL = Array{HyperbolicLayer, 2}(undef, depth, depthHL)

    # Create layers
    for j = 1:depth
        AN[j] = ActNorm(n_in; logdet=logdet)
        for k = 1:depthHL
            CL[j, k] = HyperbolicLayer(1, 1, Int(n_in/2), batchsize, 1, 1, 0; action="same", α=1f0, hidden_factor=Int(n_hidden/n_in))
        end
    end

    return NetworkHyperbolic1D(AN, CL, logdet, false)
end

# Forward pass and compute logdet
function forward(X, H::NetworkHyperbolic1D; logdet=nothing)
    isnothing(logdet) ? logdet = (H.logdet && ~H.is_reversed) : logdet = logdet

    depth, depthHL = size(H.CL)
    logdet_ = 0f0
    for j=1:depth
        logdet ? (X, logdet1) = H.AN[j].forward(X) : X = H.AN[j].forward(X)
        X_prev, X_curr = tensor_split(X)
        for k = 1:depthHL
            X_prev, X_curr = H.CL[j, k].forward(X_prev, X_curr)
        end
        X = tensor_cat(X_prev, X_curr)
        logdet && (logdet_ += logdet1)
    end
    logdet ? (return X, logdet_) : (return X)
end

# Inverse pass and compute gradients
function inverse(Z, H::NetworkHyperbolic1D; logdet=nothing)
    isnothing(logdet) ? logdet = (H.logdet && H.is_reversed) : logdet = logdet

    depth, depthHL = size(H.CL)
    logdet_ = 0f0
    for j = depth:-1:1
        Z_curr, Z_new = tensor_split(Z)
        for k = depthHL:-1:1
            Z_curr, Z_new = H.CL[j, k].inverse(Z_curr, Z_new)
        end
        Z = tensor_cat(Z_curr, Z_new)
        logdet ? (Z, logdet2) = H.AN[j].inverse(Z; logdet=true) : Z = H.AN[j].inverse(Z; logdet=false)
        logdet && (logdet_ += (logdet1 + logdet2))
    end
    logdet ? (return Z, logdet_) : (return Z)
end

# Backward pass and compute gradients
function backward(ΔZ, Z, H::NetworkHyperbolic1D)
    depth, depthHL = size(H.CL)
    for j = depth:-1:1
        ΔZ_curr, ΔZ_new = tensor_split(ΔZ)
        Z_curr, Z_new = tensor_split(Z)
        for k = depthHL:-1:1
            ΔZ_curr, ΔZ_new, Z_curr, Z_new = H.CL[j, k].backward(ΔZ_curr, ΔZ_new, Z_curr, Z_new)
        end
        ΔZ = tensor_cat(ΔZ_curr, ΔZ_new)
        Z = tensor_cat(Z_curr, Z_new)
        ΔZ, Z = H.AN[j].backward(ΔZ, Z)
    end
    return ΔZ, Z
end

# Clear gradients
function clear_grad!(H::NetworkHyperbolic1D)
    depth, depthHL = size(H.CL)
    for j=1:depth
        clear_grad!(H.AN[j])
        for k = 1:depthHL
            clear_grad!(H.CL[j, k])
        end
    end
end

# Get parameters
function get_params(H::NetworkHyperbolic1D)
    depth, depthHL = size(H.CL)
    p = []
    for j=1:depth
        p = cat(p, get_params(H.AN[j]); dims=1)
        for k = 1:depthHL
            p = cat(p, get_params(H.CL[j, k]); dims=1)
        end
    end
    return p
end

# Set is_reversed flag in full network tree
function tag_as_reversed!(H::NetworkHyperbolic1D, tag::Bool)
    depth, depthHL = size(H.CL)
    H.is_reversed = tag
    for j=1:depth
        tag_as_reversed!(H.AN[j], tag)
        for k = 1:depthHL
            tag_as_reversed!(H.CL[j, k], tag)
        end
    end
    return H
end
