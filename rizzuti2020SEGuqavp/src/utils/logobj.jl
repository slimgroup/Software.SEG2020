### Log-likehood & log-prior


export negLogLikelihood_gauss, negLogPost, negLogPrior_gauss, negLogPrior_L1smooth


function negLogLikelihood_gauss(
	d::Data, m::Model, σ2_dat::Float32,
	F::FunctionMod2Dat;
	compute_grad::Bool = false)

	# Compute functional
	Fm = F.eval(m; store_extracomp = compute_grad)
	Δd = Fm-d
	f = 1f0/(2f0*σ2_dat)*norm(Δd).^2f0

	!compute_grad ? (return f) : (return f, F.jac_adj(m, Δd)/σ2_dat)

end


function negLogPrior_gauss(
	m::Model, σ2_mod::Float32;
	b::Union{Nothing, Model} = nothing, m0::Model = Model(ModelGeom(0, 0f0), zeros(Float32, 1)), D::LinFunctionMod2Mod = LinFunctionMod2Mod(m->m, m->m),
	compute_grad::Bool = false)

	# Compute functional
	(b == nothing) ? (r = D.eval(m-m0)) : (r = D.eval(m-m0)-b)
	f = 1f0/(2f0*σ2_mod)*norm(r).^2f0

	if !compute_grad
		return f

	# Compute gradient
	else
		return f, D.eval_adj(r)/σ2_mod
	end

end


function negLogPrior_L1smooth(
	m::Model, σ_mod::Float32, ϵ2::Float32;
	m0::Model = Model(ModelGeom(0, 0f0), zeros(Float32, 1)), D::LinFunctionMod2Mod = LinFunctionMod2Mod(m->m, m->m),
	compute_grad::Bool = false)

	# Compute functional
	Dm = D.eval(m-m0)
	w = sqrt.(abs.(Dm.m).^2f0.+ϵ2)
	f = 1f0/σ_mod*vec(sum(w; dims = 1))

	if !compute_grad
		return f

	# Compute gradient
	else
		return f, D.eval_adj(Dm/w)/σ_mod
	end

end


function negLogPost(
	m::Model,
	negLogLikelihood_fun::Function, negLogPrior_fun::Function;
	compute_grad::Bool = false)

	outlike = negLogLikelihood_fun(m; compute_grad = compute_grad)
	outpr   = negLogPrior_fun(m; compute_grad = compute_grad)
	if !compute_grad
		return outlike.+outpr
	else
		return outlike[1].+outpr[1], outlike[2]+outpr[2]
	end

end
