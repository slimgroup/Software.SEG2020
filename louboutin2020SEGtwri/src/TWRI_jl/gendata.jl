################################################################################
#
# Routines for synthetic data generation
#
################################################################################



export gendata



### Wrapper for JUDI data types


function gendata(
	m::Modelall, wav::judiVector, rcv_geom::Geometry;
	opt = Options(),
	dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	save_wavefield::Bool = false)

	return gendata_rawinput(m, wav.geometry, rcv_geom, wav.data, 1:wav.nsrc; opt = opt, dt_comp = dt_comp, save_wavefield = save_wavefield)

end



### Objective wrapper for array input


function gendata(
	m::Array{R, 2}, n::NTuple{2, Int64}, d::NTuple{2, R}, o::NTuple{2, R},
	wav::judiVector, rcv_geom::Geometry;
	opt = Options(),
	dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	save_wavefield::Bool = false)

	return gendata(Model(n, d, o, m), wav, rcv_geom; opt = opt, dt_comp = dt_comp, save_wavefield = save_wavefield)

end



### Serial/parallel implementation


## Serial implementation

function gendata_rawinput(
	model_full::Modelall,
	src_geom::Geometry, rcv_geom::Geometry,
	src_data::Array{R, 2},;
	opt = Options(),
	dt_comp::Union{Nothing, R} = nothing,
	save_wavefield::Bool = false)

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
	return gendata_jldevito(model_py, model.o, src_geom, rcv_geom, src_data; opt = opt, dt_comp = dt_comp, save_wavefield = save_wavefield)

end


## Parallel implementation

function gendata_rawinput(
	model::Modelall,
	src_geom::Geometry, rcv_geom::Geometry,
	src_data,
	src_idx::UnitRange{Int64};
	opt = Options(),
	dt_comp::Union{Nothing, Array{Any, 1}} = nothing,
	save_wavefield::Bool = false)

	# Setup parallelization
	p = default_worker_pool()
	time_modeling_par = remote(gendata_rawinput)
	time_modeling = retry(time_modeling_par)

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
			if dt_comp == nothing
				dt_comp_loc = nothing
			else
				dt_comp_loc = dt_comp[j]
			end

			# Local result
			results[j] = @spawn gendata_rawinput(model, src_geom_loc, rcv_geom_loc, src_data_loc, opt = opt_loc; dt_comp = dt_comp_loc, save_wavefield = save_wavefield)

		end
	end

	# Aggregating results
	argout = Array{Array, 1}(undef, nsrc)
	for j = 1:nsrc
		argout[j] = fetch(results[j])
	end
	if save_wavefield
		return argout
	else
		return judiVector(rcv_geom, argout)
	end

end



### Julia/devito implementation


function gendata_jldevito(
	model_py::PyCall.PyObject,
	origin,
	src_geom::Geometry, rcv_geom::Geometry,
	src_data::Array{R, 2};
	opt = Options(),
	dt_comp::Union{Nothing, R} = nothing,
	save_wavefield::Bool = false)

	# Loading python modules for devito implementation of objTWRIdual
	devitomod = load_codegen_forward()

	# Interpolate input data to computational grid
	if dt_comp == nothing
		dt_comp = R(model_py.critical_dt)
	end
	q_in = time_resample(src_data, src_geom, dt_comp)[1]
	nt_comp = size(q_in, 1)
	nt_rcv = Int(trunc(rcv_geom.t[1]/dt_comp+1))

	# Set up coordinates with devito dimensions
	src_coords = setup_grid(src_geom, model_py.shape)
	rcv_coords = setup_grid(rcv_geom, model_py.shape)

	# Computing output
	if !save_wavefield
		argout = pycall(devitomod.forward,
						Tuple{Array{R, 2}, Nothing},
						model_py,
						PyReverseDims(copy(transpose(src_coords))), PyReverseDims(copy(transpose(rcv_coords))),
						PyReverseDims(copy(transpose(q_in))),
						dt_comp, SPACE_ORDER, save_wavefield)
		return argout[1]
	else
		argout = pycall(devitomod.forward,
						Tuple{Array{R, 2}, PyObject},
						model_py,
						PyReverseDims(copy(transpose(src_coords))), PyReverseDims(copy(transpose(rcv_coords))),
						PyReverseDims(copy(transpose(q_in))),
						dt_comp, SPACE_ORDER, save_wavefield)
		return argout[2].data[:, model_py.nbpml+1:end-model_py.nbpml, model_py.nbpml+1:end-model_py.nbpml]
	end

end



# Loader for python devito modeling
function load_codegen_forward()

	pushfirst!(PyVector(pyimport("sys")."path"), string(pwd(), "/src/TWRI_py/"))
	return pyimport("twri_propagators")

end
