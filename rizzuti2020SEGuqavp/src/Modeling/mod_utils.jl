### Utilities


export compute_critical_dt, compute_abcsize


function extend(m::Model, abc_geom::ABC_Geom; opt_rep::Bool = true)

	return extend(m.m, abc_geom; opt_rep = opt_rep)

end

function extend(m::Array{Float32, 2}, abc_geom::ABC_Geom; opt_rep::Bool = true) # Array size: nz, nb

	# Initialize
	m_ext = Array{Float32, 2}(undef, size(m, 1)+sum(abc_geom.size), size(m, 2))

	# Setting values on regular domain
	m_ext[abc_geom.size[1]+1:end-abc_geom.size[2], :] .= m

	# Setting values on extended domain
	if opt_rep
		m_ext[1:abc_geom.size[1],         :] .= m[1:1,     :]
		m_ext[end-abc_geom.size[2]+1:end, :] .= m[end:end, :]
	else
		m_ext[1:abc_geom.size[1],         :] .= 0f0
		m_ext[end-abc_geom.size[2]+1:end, :] .= 0f0
	end

	return m_ext

end


function taper(abc_geom::ABC_Geom)

	# Initialize
	tap_fcn = Array{Float32, 1}(undef, abc_geom.nz+sum(abc_geom.size))

	# Values on regular domain
	tap_fcn[abc_geom.size[1]+1:end-abc_geom.size[2]] .= 0f0

	# Values on extended domain
	tap_fcn[1:abc_geom.size[1]] = range(-1f0, stop = 0f0, length = abc_geom.size[1]).^2f0
	tap_fcn[end-abc_geom.size[2]+1:end] = range(0f0, stop = 1f0, length = abc_geom.size[2]).^2f0

	return tap_fcn*abc_geom.fact

end


function compute_critical_dt(m::Model, dat_pars::DataParams)

	return 1.73f0/2f0*m.geom.dz*sqrt(min(m.m...)-max(dat_pars.p...)^2f0)

end


function compute_abcsize(m::Model, dat_pars::DataParams, freq::Float32)

	return Int64(ceil(1f0/(sqrt(min(m.m...)-max(dat_pars.p...)^2f0)*freq*m.geom.dz)))

end
