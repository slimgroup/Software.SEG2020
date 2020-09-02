### Modeling operators


export seismod_fun


## Wrapper for FunctionMod2Dat type

function seismod_fun(dat_pars::DataParams, src_pars::SrcParams; abc_geom::ABC_Geom = ABC_Geom(), acc::Int64 = 4)

	return FunctionMod2Dat(
		(m; store_extracomp = false) -> forward_FD(m, dat_pars, src_pars; abc_geom = abc_geom, acc = acc, store_wav = store_extracomp),
		(m, Δd) -> jacobian_adjoint_FD(m, Δd, dat_pars, src_pars; abc_geom = abc_geom, acc = acc),
		(m, Δd) -> backward_FD(m, Δd, dat_pars, src_pars; abc_geom = abc_geom, acc = acc))

end


## Forward & backward (finite-differences)

function forward_FD(
	m::Model, dat_pars::DataParams, src_pars::SrcParams;
	abc_geom::ABC_Geom = ABC_Geom(), acc::Int64 = 4, store_wav::Bool = false)

	# Extending domain to include ABC
	nz, nb = size(m)
	m_ext = reshape(extend(m.m, abc_geom), 1, :, nb)
	tap = reshape(taper(abc_geom), 1, :)
	nz_ext = length(tap)

	# Initialize output
	nt, np = size(dat_pars)
	dat = Array{Float32, 3}(undef, nt, np, nb)
	if store_wav
		dttu = Array{Float32, 4}(undef, nt, np, nz_ext, nb)
	end

	# Initial conditions
	u0 = zeros(Float32, np, nz_ext, nb)
	u1 = zeros(Float32, np, nz_ext, nb)

	# Loop over time
	dz = m.geom.dz
	dt = dat_pars.dt
	wav = src_pars.wav
	abc_size = abc_geom.size
	p = dat_pars.p
	@inbounds for idx_t = 1:nt

		# Time step
		u2 = ((2f0*u1-u0).*(m_ext.-p.^2f0)/dt^2f0.+Dzz(u1, dz; acc = acc).+tap/dt.*u1)./((m_ext.-p.^2f0)/dt^2f0.+tap/dt)

		# Source injection
		isrc1 = Int64(floor(src_pars.zsrc/dz))+1
		u2[:, abc_size[1]+isrc1, :] += wav[idx_t]./((m_ext[:, abc_size[1]+1, :].-p.^2f0)/dt^2f0.+tap[:, abc_size[1]+1]/dt)

		# Float32ecording wavefield at surface position
		dat[idx_t, :, :] = reshape(u1[:, abc_size[1]+1, :], np, nb)

		# Float32ecording full wavefield
		if store_wav
			dttu[idx_t, :, :, :] = ((u2-2f0*u1+u0)/dt^2f0)
		end

		# Update wavefields
		u0 = u1
		u1 = u2

	end

	if !store_wav
		return Data(dat, dat_pars, nothing)
	else
		return Data(dat, dat_pars, dttu)
	end

end

function backward_FD(
	m::Model, Δd::Data,
	dat_pars::DataParams, src_pars::SrcParams;
	abc_geom::ABC_Geom = ABC_Geom(), acc::Int64 = 4)

	# Extending domain to include ABC
	nz, nb = size(m)
	m_ext = reshape(extend(m.m, abc_geom), 1, :, nb)
	tap = reshape(taper(abc_geom), 1, :)
	nz_ext = length(tap)

	# Initialize output
	nt, np = size(dat_pars)
	q = zeros(Float32, np, nz, nb)

	# Final conditions
	v0 = zeros(Float32, np, nz_ext, nb)
	v1 = zeros(Float32, np, nz_ext, nb)

	# Loop over time
	dz = m.geom.dz
	dt = dat_pars.dt
	wav = src_pars.wav
	abc_size = abc_geom.size
	p = dat_pars.p
	@inbounds for idx_t = nt:-1:1

		# Time step
		v2 = ((2f0*v1-v0).*(m_ext.-p.^2f0)/dt^2f0.+Dzz(v1, dz; acc = acc).+tap/dt.*v1)./((m_ext.-p.^2f0)/dt^2f0.+tap/dt)

		# Source injection
		v2[:, abc_size[1]+1, :] += Δd.d[idx_t, :, :]./((m_ext[:, abc_size[1]+1, :].-p.^2f0)/dt^2f0.+tap[abc_size[1]+1]/dt)

		# Update source
		q += v2[:, abc_size[1]+1:end-abc_size[2], :]*wav[idx_t]

		# Update wavefields
		v0 = v1
		v1 = v2

	end

	return q

end


## Jacobian adjoint

function jacobian_adjoint_FD(
	m::Model, Δd::Data,
	dat_pars::DataParams, src_pars::SrcParams;
	abc_geom::ABC_Geom = ABC_Geom(), acc::Int64 = 4)

	# Initialize output
	nz, nb = size(m)
	Δm = zeros(Float32, nz, nb)

	# Float32epeat forward computation, if not previously computed
	if Δd.dttu == nothing
		Δd.dttu = forward_FD(m, dat_pars, src_pars; abc_geom = abc_geom, acc = acc, store_wav = true).dttu
	end

	# Extending domain to include ABC
	nz, nb = size(m)
	m_ext = reshape(extend(m.m, abc_geom), 1, :, nb)
	tap = reshape(taper(abc_geom), 1, :)
	nz_ext = length(tap)

	# Final conditions
	nt, np = size(dat_pars)
	v0 = zeros(Float32, np, nz_ext, nb)
	v1 = zeros(Float32, np, nz_ext, nb)

	# Loop over time
	dz = m.geom.dz
	dt = dat_pars.dt
	wav = src_pars.wav
	abc_size = abc_geom.size
	p = dat_pars.p
	@inbounds for idx_t = nt:-1:1

		# Time step
		v2 = ((2f0*v1-v0).*(m_ext.-p.^2f0)/dt^2f0.+Dzz(v1, dz; acc = acc).+tap/dt.*v1)./((m_ext.-p.^2f0)/dt^2f0.+tap/dt)

		# Source injection
		v2[:, abc_size[1]+1, :] += Δd.d[idx_t, :, :]./((m_ext[:, abc_size[1]+1, :].-p.^2f0)/dt^2f0.+tap[abc_size[1]+1]/dt)

		# Update gradient
		Δm_ = -reshape(sum(v1.*Δd.dttu[idx_t, :, :, :], dims = 1), :, nb)
		Δm[1,       :] += vec(sum(Δm_[1:abc_size[1]+1,                 :], dims = 1))
		Δm[2:end-1, :] +=         Δm_[abc_size[1]+2:end-abc_size[2]-1, :]
		Δm[end,     :] += vec(sum(Δm_[end-abc_size[2]:end,             :], dims = 1))

		# Update wavefields
		v0 = v1
		v1 = v2

	end

	return Model(m.geom, Δm)

end
