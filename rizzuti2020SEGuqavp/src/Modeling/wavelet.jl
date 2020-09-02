### Wavelet


export cfreq_wavelet


function cfreq_filt(nt::Int64, dt::Float32, cfreqs::NTuple{4, Float32}; type::String = "bandpass", edge::Function = hanning, flag_fft::Bool = true)

	# Initialization
	Filter = zeros(ComplexF32, nt)

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
	Filter[1:i1-1] .= 0f0
	k = collect(i1:i2-1); Filter[k] .= (k.-i1)./(i2-i1)
	Filter[i2:i3] .= 1f0
	k = collect(i3+1:i4); Filter[k] .= 1f0.-(k.-i3)./(i4-i3)
	Filter[i4+1:end] .= 0f0
	SL = Int64((nt-1)/2)
	Filter[SL+2:end] = Filter[SL+1:-1:2]

	# Zeroing edge values in time domain
	filter = ifftshift(Float32.(edge(length(Filter))).*fftshift(real.(ifft(Filter))))

	# Final Fourier transform
	if flag_fft
		return fft(filter)
	else
		return filter
	end

end


function cfreq_wavelet(SL::Int64, nt::Int64, dt::Float32, cfreqs::NTuple{4, Float32}; type::String = "bandpass", edge::Function = hanning)

	return [fftshift(cfreq_filt(2*SL+1, dt, cfreqs; type = type, edge = edge, flag_fft = false)); zeros(Float32, nt-(2*SL+1))]

end
