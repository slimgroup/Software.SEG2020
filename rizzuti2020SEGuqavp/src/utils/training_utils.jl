### Training utilities


export training!


function training!(T::InvertibleNetwork, fval::Array{Float32, 1}, loss::Function, Ztrain::Array{Float32, 4}, nepochs::Int64, batchsize::Int64, opt::Optimiser; verbose::Bool = true, verbose_b::Bool = false)

	trainsize = size(Ztrain, 4)
	nbatches = Int64(round(trainsize/batchsize))

	# Loop over iterations
	for n = 1:nepochs

	    # Shuffling data
	    idx_train_perm = randperm(trainsize)

	    # Loop over batches
	    for idx_batch = 1:nbatches

	        idx = idx_train_perm[(idx_batch-1)*batchsize+1:idx_batch*batchsize]
	        Z = Ztrain[:, :, :, idx]

	        # Evaluate objective and backpropagating
	        l = loss(Z)
	        append!(fval, l)

	        # Update network
	        for p in get_params(T)
	            update!(opt, p.data, p.grad)
	        end

	        # Clear gradients
	        clear_grad!(T)

		    # Print msg
		    verbose_b && println("         Batch: [", idx_batch, "/", nbatches, "] --- fval = ", fval[end])

	    end

	    # Print msg
	    verbose && println("Epoch: [", n, "/", nepochs, "] --- fval = ", fval[end])

	end

	return fval

end
