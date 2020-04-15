################################################################################
#
# Routines for the the evaluation and gradient computation of FWI
#
################################################################################



export objFWI, objFWI!, objFWI_rawinput



### Optim.jl-adapted wrapper


function objFWI!(
	F, Gm, m::Array{R, 2},
	n::NTuple{2, Int64}, d::NTuple{2, R}, o::NTuple{2, R},
	wav::judiVector,
	dat::judiVector;
	opt = Options(space_order=SPACE_ORDER),
	dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	Filter::Union{Nothing, Array{C, 1}, Array{Array{C, 1}, 1}} = nothing,
	gradmprec_fun = g->g)

	if Gm == nothing
		mode = "eval"
	else
		mode = "grad"
	end
	argout = objFWI(m, n, d, o, wav, dat; opt = opt, mode = mode, dt_comp = dt_comp, Filter = Filter, gradmprec_fun = gradmprec_fun)
	if Gm != nothing
		fval = argout[1]
		println(size(Gm), " ", size(argout[2]))
		Gm .= argout[2]
	else
		fval = argout
	end
	return fval

end

function objFWI!(
	F, Gm, m::Array{R, 2},
	model::Modelall,
	wav::judiVector,
	dat::judiVector;
	opt = Options(space_order=SPACE_ORDER),
	dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	Filter::Union{Nothing, Array{C, 1}, Array{Array{C, 1}, 1}} = nothing,
	gradmprec_fun = g->g)

	if Gm == nothing
		mode = "eval"
	else
		mode = "grad"
	end
	model.m = m
	argout = objFWI(m, model, wav, dat; opt = opt, mode = mode, dt_comp = dt_comp, Filter = Filter, gradmprec_fun = gradmprec_fun)
	if Gm != nothing
		fval = argout[1]
		Gm .= argout[2]
	else
		fval = argout
	end
	return fval

end

### Objective wrapper for JUDI data types


function objFWI(
	m::Modelall,
	wav::judiVector,
	dat::judiVector;
	opt = Options(space_order=SPACE_ORDER),
	mode = "eval", dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	Filter::Union{Nothing, Array{C, 1}, Array{Array{C, 1}, 1}} = nothing,
	gradmprec_fun = g->g)
	return objFWI_rawinput(m, wav.geometry, dat.geometry, wav.data, dat.data, 1:length(dat.data); opt = opt, mode = mode, dt_comp = dt_comp, Filter = Filter, gradmprec_fun = gradmprec_fun)
end



### Objective wrapper for array input


function objFWI(
	m::Array{R, 2}, n::NTuple{2, Int64}, d::NTuple{2, R}, o::NTuple{2, R},
	wav::judiVector, dat::judiVector;
	opt = Options(space_order=SPACE_ORDER),
	mode = "eval", dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	Filter::Union{Nothing, Array{C, 1}, Array{Array{C, 1}, 1}} = nothing,
	gradmprec_fun = g->g)

	return objFWI(Model(n, d, o, m), wav, dat; opt = opt, mode = mode, dt_comp = dt_comp, Filter = Filter, gradmprec_fun = gradmprec_fun)

end



function objFWI(
	m::Array{R, 2}, model::Modelall,
	wav::judiVector, dat::judiVector;
	opt = Options(space_order=SPACE_ORDER),
	mode = "eval", dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	Filter::Union{Nothing, Array{C, 1}, Array{Array{C, 1}, 1}} = nothing,
	gradmprec_fun = g->g)
	model.m = m
	return objFWI(model, wav, dat; opt = opt, mode = mode, dt_comp = dt_comp, Filter = Filter, gradmprec_fun = gradmprec_fun)

end

### Objective: serial/parallel implementation


## Serial implementation

function objFWI_rawinput(
	model_full::Modelall,
	src_geom::Geometry, rcv_geom::Geometry,
	src_data::Array{R, 2}, dat_data::Array{R, 2};
	opt = Options(space_order=SPACE_ORDER),
	mode = "eval", dt_comp::Union{Nothing, R} = nothing,
	Filter::Union{Nothing, Array{C, 1}} = nothing,
	gradmprec_fun = g->g)

	# Setting pre-defined absorbing layer size
	model_full.nb = NB

	# Load full geometry for out-of-core geometry containers
	typeof(rcv_geom) == GeometryOOC && (rcv_geom = Geometry(rcv_geom))
	typeof(src_geom) == GeometryOOC && (src_geom = Geometry(src_geom))
	length(model_full.n) == 3 ? dims = [3, 2, 1] : dims = [2, 1] # model dimensions for Python are (z,y,x) and (z,x)

	# Limit model to area with sources/receivers
	if opt.limit_m == true
		model = deepcopy(model_full)
		model = limit_model_to_receiver_area(src_geom, rcv_geom, model, opt.buffer_size)
	else
		model = model_full
	end

	# Set up Python model structure
	model_py = devito_model(model, "F", 1, opt, 0)

	# Remove receivers outside the modeling domain (otherwise leads to segmentation faults)
	rcv_geom = remove_out_of_bounds_receivers(rcv_geom, model)

	# Call to objective with Julia/Devito interface function
	return objFWI_jldevito(model_py, model.o, src_geom, rcv_geom, src_data, dat_data; opt = opt, mode = mode, dt_comp = dt_comp, Filter = Filter, gradmprec_fun = gradmprec_fun)

end


## Parallel implementation

function objFWI_rawinput(
	model::Modelall,
	src_geom::Geometry, rcv_geom::Geometry,
	src_data, dat_data,
	src_idx::UnitRange{Int64};
	opt = Options(space_order=SPACE_ORDER),
	mode = "eval", dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	Filter::Union{Nothing, Array{C, 1}, Array{Array{C, 1}, 1}} = nothing,
	gradmprec_fun = g->g)

	# Setup parallelization
	p = default_worker_pool()
	objFWI_rawinput_par = remote(TWRIdual.objFWI_rawinput)
	objFWI_rawinput = retry(objFWI_rawinput_par)

	# Initialize output
	nsrc = length(src_idx)
	results = Array{Any}(undef, nsrc)

	# Process shots from source channel asynchronously
	@sync begin
		for j = 1:nsrc

			# Local geometry for current position
			opt_loc = subsample(opt, j)
			src_geom_loc = subsample(src_geom, j)
			rcv_geom_loc = subsample(rcv_geom, j)

			# Selecting variables for current shot index
			src_data_loc = src_data[j]
			dat_data_loc = dat_data[j]
			if isa(Filter, Array{Array{C, 1}, 1})
				Filter_loc = Filter[j]
			else
				Filter_loc = Filter
			end
			if dt_comp == nothing
				dt_comp_loc = nothing
			else
				dt_comp_loc = dt_comp[j]
			end

			# Local result
			results[j] = @spawn objFWI_rawinput(model, src_geom_loc, rcv_geom_loc, src_data_loc, dat_data_loc, opt = opt_loc; mode = mode, dt_comp = dt_comp_loc, Filter = Filter_loc, gradmprec_fun = gradmprec_fun)

		end
	end

	# Aggregating results
	if mode == "eval"
		argout = fetch(results[1])
		for j = 2:nsrc
			argout += fetch(results[j])
		end
	elseif mode == "grad"
		arg = fetch(results[1])
		argout1 = arg[1]
		argout2 = arg[2]
		for j = 2:nsrc
			arg = fetch(results[j])
			argout1 += arg[1]
			argout2 += arg[2]
		end
		argout = (argout1, argout2)
	end
	return argout./nsrc

end



### Julia/devito call


function objFWI_jldevito(
	model_py::PyCall.PyObject,
	origin,
	src_geom::Geometry, rcv_geom::Geometry,
	src_data::Array{R, 2}, dat_data::Array{R, 2};
	opt = Options(space_order=SPACE_ORDER),
	mode = "eval", dt_comp::Union{Nothing, R} = nothing,
	Filter::Union{Nothing, Array{C, 1}} = nothing,
	gradmprec_fun = g->g)
	# Interpolate input data to computational grid
	if dt_comp == nothing
		dt_comp = R(model_py.critical_dt)
	end
	q_in = time_resample(src_data, src_geom, dt_comp)[1]
	dat_in = time_resample(dat_data, rcv_geom, dt_comp)[1]
	nt_comp = size(q_in, 1)
	nt_rcv = Int(trunc(rcv_geom.t[1]/dt_comp+1))

	# Set up coordinates with devito dimensions
	src_coords = setup_grid(src_geom, model_py.shape)
	rcv_coords = setup_grid(rcv_geom, model_py.shape)

	fwi_ops = load_codegen_FWI(model_py)
	# Computing output
	if mode == "eval"
		argout = pycall(fwi_ops.fwi_fun, R, src_coords, rcv_coords, q_in, dat_in, Filter, mode)
	elseif mode == "grad"
		argout = pycall(fwi_ops.fwi_fun, Tuple{R, Array{R, 2}}, src_coords, rcv_coords, q_in, dat_in, Filter, mode)
	end

	# Output post-processing
	if mode == "grad"
		gm = remove_padding(argout[2], model_py.nbl, true_adjoint = opt.sum_padding) # remove AB layers
		argout = (argout[1], gradmprec_fun(gm))
	end

	return argout

end


# Loader for python devito modeling
@cache function load_codegen_FWI(model)

	pushfirst!(PyVector(pyimport("sys")."path"), string(pwd(), "/src/TWRI_py/"))
	int = pyimport("wri_interface")
	return int.WRIInterface(model, SPACE_ORDER)
end
