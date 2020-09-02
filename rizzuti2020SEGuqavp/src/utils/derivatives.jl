### Derivative routines


export derivative_linop


## Wrapper for derivative operator

function derivative_linop(m_geom::ModelGeom)

	return LinFunctionMod2Mod(
		m -> Model(m_geom, Dz(    m.m, m_geom.dz)),
		m -> Model(m_geom, Dz_adj(m.m, m_geom.dz)))

end


## First order

function Dz(m::Array{Float32, 2}, dz::Float32)

		Dz_m = Array{Float32, 2}(undef, size(m))
		Dz_m[1:end-1, :]  = (m[2:end, :]-m[1:end-1, :])/dz
		Dz_m[end,     :] .= 0f0

		return Dz_m

end

function Dz_adj(m::Array{Float32, 2}, dz::Float32)

		Dz_m = Array{Float32, 2}(undef, size(m))
		Dz_m[1,       :] .= (m[1,       :]              )/dz
		Dz_m[2:end-1, :] .= (m[2:end-1, :]-m[1:end-2, :])/dz
		Dz_m[end,     :] .= (             -m[end-1,   :])/dz

		return -Dz_m

end


## Second order

function Dzz(u::Array{Float32, 3}, dz::Float32; acc::Int64 = 2) # Array size: np, nz, nb

		if acc == 2
			return Dzz2(u, dz)
		elseif acc == 4
			return Dzz4(u, dz)
		end

end

function Dzz2(u::Array{Float32, 3}, dz::Float32)

	Dzz_u = Array{Float32, 3}(undef, size(u))
	Dzz_u[:, 1,       :] = (                -2f0*u[:, 1,       :]+u[:, 2,     :])/dz^2f0
	Dzz_u[:, 2:end-1, :] = (u[:, 1:end-2, :]-2f0*u[:, 2:end-1, :]+u[:, 3:end, :])/dz^2f0
	Dzz_u[:, end,     :] = (u[:, end-1,   :]-2f0*u[:, end,     :]               )/dz^2f0

	return Dzz_u

end

function Dzz4(u::Array{Float32, 3}, dz::Float32)

	Dzz_u = Array{Float32, 3}(undef, size(u))
	Dzz_u[:, 1,       :] = (                                                 -5f0/2f0*u[:, 1,       :]+4f0/3f0*u[:, 2,       :]-1f0/12f0*u[:, 3,     :])/dz^2f0
	Dzz_u[:, 2,       :] = (                          4f0/3f0*u[:, 1,       :]-5f0/2f0*u[:, 2,       :]+4f0/3f0*u[:, 3,       :]-1f0/12f0*u[:, 4,     :])/dz^2f0
	Dzz_u[:, 3:end-2, :] = (-1f0/12f0*u[:, 1:end-4, :]+4f0/3f0*u[:, 2:end-3, :]-5f0/2f0*u[:, 3:end-2, :]+4f0/3f0*u[:, 4:end-1, :]-1f0/12f0*u[:, 5:end, :])/dz^2f0
	Dzz_u[:, end-1,   :] = (-1f0/12f0*u[:, end-3,   :]+4f0/3f0*u[:, end-2,   :]-5f0/2f0*u[:, end-1,   :]+4f0/3f0*u[:, end,     :]                       )/dz^2f0
	Dzz_u[:, end,     :] = (-1f0/12f0*u[:, end-2,   :]+4f0/3f0*u[:, end-1,   :]-5f0/2f0*u[:, end,     :]                                               )/dz^2f0

	return Dzz_u

end
