# Invertible network utilities
# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: August 2020

export get_pars, get_grads, set_pars!, tmapz2x_den, initialize_id!

function get_pars(H::InvertibleNetwork)
    θ = get_params(H)
    θvec = Array{Array{Float32}, 1}(undef, length(θ))
    for i = 1:length(θ)
        θvec[i] = θ[i].data
    end
    return θvec
end

function set_pars!(H::InvertibleNetwork, θ::Array)
    θH = get_params(H)
    for i = 1:length(θ)
        isa(θ[i], Parameter) ? θH[i].data = copy(θ[i].data) : θH[i].data = copy(θ[i])
    end
end

function get_grads(H::InvertibleNetwork)
    θ = get_params(H)
    θvec = Array{Array{Float32}, 1}(undef, length(θ))
    for i = 1:length(θ)
        θvec[i] = θ[i].grad
    end
    return θvec
end

function tmapz2x_den(T::InvertibleNetwork, X::Array{Float32, 4})
    Z = T.inverse(X)
    logdet = Array{Float32, 4}(undef, 1, 1, 1, size(X, 4))
    for i = 1:size(X, 4)
        _, logdet[i] = T.forward(Z[:, :, :, i:i])
    end
    return neglogden = 0.5f0*sum(abs.(Z).^2f0, dims = 3)+logdet
end

function initialize_id!(H::NetworkHINT1D; αw::Float32 = 0f0, αb::Float32 = 0f0)
    n = length(H.CL)
    m = length(H.CL[1].CL)
    for i = 1:n
        for j = 1:m
            # H.CL[i].CL[j].RB.W1.data .*= αw
            # H.CL[i].CL[j].RB.W2.data .*= αw
            H.CL[i].CL[j].RB.W3.data .*= 0f0
            # H.CL[i].CL[j].RB.b1.data .*= αb
            # H.CL[i].CL[j].RB.b2.data .*= αb
       end
   end
   # H.CL[n].CL[1].RB.W1.data .= 0f0
   # H.CL[n].CL[1].RB.W2.data .= 0f0
   # H.CL[n].CL[1].RB.W3.data .= 0f0
end

function initialize_id!(H::NetworkGlow1D; αw::Float32 = 0f0, αb::Float32 = 0f0)
    n = length(H.CL)
    for i = 1:n
        # H.CL[i].RB.W1.data .*= αw
        # H.CL[i].RB.W2.data .*= αw
        H.CL[i].RB.W3.data .*= 0f0
        # H.CL[i].RB.b1.data .*= αb
        # H.CL[i].RB.b2.data .*= αb
   end
   # H.CL[n].CL[1].RB.W1.data .= 0f0
   # H.CL[n].CL[1].RB.W2.data .= 0f0
   # H.CL[n].CL[1].RB.W3.data .= 0f0
end
