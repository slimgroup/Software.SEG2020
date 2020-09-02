import Base.*, Base.sqrt
export langevinSampler, IdLangevin



struct IdLangevin end
function *(P::IdLangevin, x)
    return x
end
function sqrt(P::IdLangevin)
    return IdLangevin()
end


function langevinSampler(
    neglogprob::Function,
    nsamples::Int64, x0::Array{Float32}, ϵ::Union{Float32, NTuple{3, Float32}};
    flag_MALA::Bool = false,
    P::Union{IdLangevin, AbstractArray{Float32, 2}} = IdLangevin(),
    verbose::Bool = true)

    # Initialize output variables
    fval = zeros(Float32, nsamples)
    samples = Array{Float32, 2}(undef, prod(size(x0)), nsamples)

    # Computing sqrt of preconditioner
    sqrtP = sqrt(P)

    # Noise parameters
    ϵ_vec = compute_ϵvec(ϵ, nsamples)

    # Starting model
    x = x0
    samples[:, 1] = x0

    # Loop over iterations
    n = 1
    accmem = 10
    accrej_array = Array{Bool, 1}(undef, accmem); accrej_array .= false
    while n < nsamples

        # Evaluate objective and gradients
        fval[n], g = neglogprob(x)

        # Print current iteration
        if verbose
            if !flag_MALA
                println("Iter: [", n, "/", nsamples, "] --- fval = ", fval[n])
            else
                println("Iter: [", n, "/", nsamples, "] --- fval = ", fval[n], ", acc_rate = ", sum(accrej_array)/accmem)
            end
        end

        # Proposed model
        Pg = P*g
        x_ = x-ϵ_vec[n]*Pg+sqrt(2*ϵ_vec[n])*(sqrtP*randn(Float32, size(x)))

        # Acceptance-rejection step
        accrej_array[1:end-1] = accrej_array[2:end]
        if flag_MALA
            neglogq = Float32(0.25/ϵ_vec[n])*norm(x_-x+ϵ_vec[n]*Pg)^2
            fval_, g_ = neglogprob(x_); Pg_ = P*g_
            neglogq_ = Float32(0.25/ϵ_vec[n])*norm(x-x_+ϵ_vec[n]*Pg_)^2
            α = min(Float32(0), -(fval_+neglogq-fval[n]-neglogq_))
            if α >= log(rand(Float32))
                x = x_
                accrej_array[end] = true
            else
                n -= 1
                accrej_array[end] = false
            end
        else
            x = x_
        end

        # Collect sample
        n += 1
        samples[:, n] = x

    end

    return fval, samples

end

function compute_ϵvec(ϵ::Union{Float32, NTuple{3, Float32}}, n::Int64)
    if isa(ϵ, NTuple{3, Float32})
        b = (n-1)/((ϵ[1]/ϵ[2])^(1/ϵ[3])-1)
        a = b^gamma*ϵ0
        return a./(b.+collect(1:n).-1).^ϵ[3]
    else
        return ϵ*ones(Float32, n)
    end
end
