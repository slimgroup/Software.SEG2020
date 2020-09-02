### Data types


import Base.+, Base.-, Base.*, Base./, Base.==, Base.size, LinearAlgebra.norm, LinearAlgebra.dot, Statistics.mean, Statistics.var
export ModelGeom, Model, DataParams, SrcParams, Data, ABC_Geom, FunctionMod2Dat, LinFunctionMod2Mod


## Model geometry

mutable struct ModelGeom
	nz::Int64
	dz::Float32
end

function ==(geom1::ModelGeom, geom2::ModelGeom)
	return (geom1.nz == geom2.nz) && (geom1.dz == geom2.dz)
end


## Model

mutable struct Model
	geom::ModelGeom
	m::Array{Float32, 2}
end

function +(m1::Model, m2::Model)
	if m1.geom == m2.geom
		return Model(m1.geom, m1.m.+m2.m)
	end
end

function +(m1::Model, m2::Array{Float32})
	return Model(m1.geom, m1.m.+m2)
end

function +(m1::Model, c::Float32)
	return Model(m1.geom, m1.m.+c)
end

function -(m1::Model, m2::Model)
	if m1.geom == m2.geom
		return Model(m1.geom, m1.m.-m2.m)
	end
end

function /(m::Model, c::Float32)
	return Model(m.geom, m.m/c)
end

function /(m1::Model, m2::Array{Float32})
	return Model(m1.geom, m1.m./m2)
end

function dot(m1::Model, m2::Model)
	if m1.geom == m2.geom
		return vec(sum(m1.m.*m2.m, dims = 1))
	end
end

function dot(m1::Model, m2::Array{Float32, 2})
	return vec(sum(m1.m.*m2, dims = 1))
end

function dot(m1::Array{Float32, 2}, m2::Model)
	return vec(sum(m1.*m2.m, dims = 1))
end

function size(m::Model)
	return size(m.m)
end

function norm(m::Model)
	return vec(sqrt.(sum(m.m.^2f0, dims = 1)))
end

function mean(m::Model)
	return Model(m.geom, mean(m.m, dims = 2))
end

function var(m::Model)
	return Model(m.geom, var(m.m, dims = 2))
end


## Data parameters

mutable struct DataParams
	nt::Int64
	dt::Float32
	p::Array{Float32, 1}
end

function ==(dat_pars1::DataParams, dat_pars2::DataParams)
	return (dat_pars1.nt == dat_pars2.nt) && (dat_pars1.dt == dat_pars2.dt) && (dat_pars1.p == dat_pars2.p)
end

function size(dat_pars::DataParams)
	return dat_pars.nt, length(dat_pars.p)
end


## Source parameters

mutable struct SrcParams
	wav::Array{Float32, 1}
	dt::Float32
	zsrc::Float32
end


## Data

mutable struct Data
	d::Array{Float32, 3}
	pars::DataParams
	dttu::Union{Nothing, Array{Float32, 4}}
end

function +(d1::Float32, d2::Data)
	return Data(d1.+d2.d, d2.pars, nothing)
end

function +(d1::Data, d2::Data)
	d1.pars == d2.pars && (return Data(d1.d.+d2.d, d1.pars, nothing))
end

function +(d1::Data, d2::Array{Float32, 3})
	return Data(d1.d.+d2, d1.pars, nothing)
end

function -(d1::Array{Float32, 3}, d2::Data)
	return Data(d1.-d2.d, d2.pars, nothing)
end

function -(d1::Data, d2::Array{Float32, 3})
	return Data(d1.d.-d2, d1.pars, nothing)
end

function -(d1::Data, d2::Data)
	return Data(d1.d.-d2.d, d1.pars, nothing)
end

function *(d1::Float32, d2::Data)
	return Data(d1*d2.d, d2.pars, d2.dttu)
end

function norm(d::Data)
	return vec(sqrt.(sum(d.d.^2f0, dims = (1, 2))))
end


## Absorbing boundary condition options

mutable struct ABC_Geom
	nz::Union{Nothing, Int64}
	size::NTuple{2, Int64}
	fact::Float32
end

function ABC_Geom(; nz::Union{Nothing, Int64} = nothing, size::NTuple{2, Int64} = (40, 40), fact::Float32 = 1f0)
	return ABC_Geom(nz, size, fact)
end


## Model-to-data functions

mutable struct FunctionMod2Dat
	eval::Function
	jac_adj::Function
	back::Function
end


## Model-to-model linear functions

mutable struct LinFunctionMod2Mod
	eval::Function
	eval_adj::Function
end
