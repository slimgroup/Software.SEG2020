################################################################################
#
# Data filtering utilities
#
################################################################################



export cfreqfilt, cfreq_wavelet, applyfilt



### Filters


function cfreqfilt(nt::Array{Any, 1}, dt::Array{Any, 1}, cfreqs::NTuple{4, R}; type::String = "bandpass", edge = hamming, flag_fft::Bool = true, norm_fact::Union{Nothing, Tuple{judiVector, NTuple{2, R}}} = nothing)

	if flag_fft
		Filter = Array{Array{C, 1}, 1}(undef, length(dt))
	else
		Filter = Array{Array{R, 1}, 1}(undef, length(dt))
	end
	if norm_fact != nothing
		fourNormDat = computeFourierNorm(nt, norm_fact[1], norm_fact[2])
	end
	for i = 1:length(dt)
		Filter[i] = cfreqfilt(nt[i], dt[i], cfreqs; type = type, edge = edge, flag_fft = true)
		if norm_fact != nothing
			Filter[i] ./= fourNormDat[i]
		end
		if !flag_fft
			Filter[i] = real.(ifft(Filter[i]))
		end
	end
	return Filter

end


function cfreqfilt(nt::Int64, dt::R, cfreqs::NTuple{4, R}; type::String = "bandpass", edge = hamming, flag_fft::Bool = true)

	# Initialization
	Filter = zeros(C, nt)

	# Index corresponding to corner frequencies
	i1 = Int64(ceil(cfreqs[1]*nt*dt)+1)
	i2 = Int64(ceil(cfreqs[2]*nt*dt)+1)
	i3 = Int64(floor(cfreqs[3]*nt*dt)+1)
	i4 = Int64(floor(cfreqs[4]*nt*dt)+1)
	if type == "lowpass"
		i1 = 1
		i2 = 1
	elseif type == "highpass"
		i3 = nt
		i4 = nt
	end

	# Populating filter in the frequency domain
	Filter[1:i1-1] .= 0
	k = collect(i1:i2-1); Filter[k] .= (k.-i1)./(i2-i1)
	Filter[i2:i3] .= 1
	k = collect(i3+1:i4); Filter[k] .= R(1).-(k.-i3)./(i4-i3)
	Filter[i4+1:end] .= 0
	SL = Int64((nt-1)/2)
	Filter[SL+2:end] = Filter[SL+1:-1:2]

	# Zeroing edge values in time domain
	filter = ifftshift(R.(edge(length(Filter))).*fftshift(real.(ifft(Filter))))

	# Final Fourier transform
	if flag_fft
		return fft(filter)
	else
		return filter
	end

end


function cfreq_wavelet(SL::Int64, nt::Int64, dt::R, cfreqs::NTuple{4, R}; type::String = "bandpass", edge = hamming, flag_fft::Bool = true)

	return [cfreqfilt(2*SL+1, dt, cfreqs; type = type, edge = edge, flag_fft = false); zeros(R, nt-(2*SL+1))]

end


function whitening_filter(dat::Array{R, 2}, dt::R, B::NTuple{2, R})

	nt = size(dat, 1)
	Dat = fft(dat, 1)
	normDat = sqrt.(sum(abs.(Dat).^2, 2))
	df = 1/(nt*dt)
	freqs = ((1:nt).-1)*df
	idxB = findall(freqs .>= B[1] .& freqs .<= B[2])
	Filter = zeros(R, nt)
	Filter[idxB] = 1f0./normDat[idxB]

end



### Filter application


function applyfilt(dat::judiVector, Filter::Union{Array{Array{C, 1}, 1}, Array{C, 1}})

	dat_filt = Array{Array, 1}(undef, dat.nsrc)
	flag_arr = isa(Filter, Array{C, 1})
	for i = 1:dat.nsrc
		if flag_arr
			dat_filt[i] = applyfilt(dat.data[i], Filter)
		else
			dat_filt[i] = applyfilt(dat.data[i], Filter[i])
		end
	end
	return judiVector(dat.geometry, dat_filt)

end


function applyfilt(dat::Array{R, 2}, Filter::Array{C, 1})

	Dat = fft(zeropadding(dat, length(Filter)), 1)
	return real.(ifft(Dat.*Filter, 1)[1:size(dat, 1), :])

end


function zeropadding(dat::Array{R, 2}, pad::Int64)

	dat_ = zeros(R, (pad, size(dat, 2)))
	dat_[1:size(dat, 1), :] .= dat
	return dat_

end


###


function computeFourierNorm(nt::Array{Any, 1}, dat::judiVector, B::NTuple{2, R}; norm_src::Bool = true)

	fourNorm = Array{Array{R, 1}, 1}(undef, dat.nsrc)
	for i = 1:dat.nsrc
		fourNorm[i] = computeFourierNorm(nt[i], dat.geometry.dt[i], dat.data[i], B)
	end
	return fourNorm

end


function computeFourierNorm(nt::Int64, dt::R, dat::Array{R, 2}, B::NTuple{2, R})

	df = 1f0/(nt*dt)
	freqs = ifftshift(R.(-(nt-1)/2:(nt-1)/2))*df
	Dat = fft(zeropadding(dat, nt), 1)
	fourNorm = vec(sqrt.(sum(abs.(Dat).^2, dims = 2)))
	fourNorm[findall(.!((abs.(freqs) .>= B[1]) .& (abs.(freqs) .<= B[2])))] .= 1f0
	return fourNorm

end
