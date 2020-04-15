################################################################################
#
# Set of utilities for TWRIdual
#
################################################################################



export judiVector2vec, vec2judiVector, model2vec, vec2model, contr2abs, abs2contr, gradprec_contr2abs, devito_model, gradient_desc

### JUDI data types/array format translations


function judiVector2vec(y::judiVector)
	return vec(cat(y.data..., dims = 2))
end


function vec2judiVector(rcv_geom::GeometryIC, y_vec::Array)
	nsrc = length(rcv_geom.xloc)
	y_arr = Array{Array}(undef, nsrc)
	for i = 1:nsrc
		nrcv = length(rcv_geom.xloc[i])
		nt = rcv_geom.nt[i]
		y_arr[i] = reshape(y_vec[nrcv*nt*(i-1)+1:nrcv*nt*i], nt, nrcv)
	end
	return judiVector(rcv_geom, y_arr)
end


function model2vec(m)
	return vec(m.m)
end


function vec2model(n, d, o, m)
	return Model(n, d, o, reshape(m, n))
end



### Utilities for model input preprocessing and gradient postprocessing


function contr2abs(xvec::Array{R, 1}, mask::BitArray{2}, mb::Array{R, 2})
# Preprocessing steps: - effective to global domain
#                      - contrast to absolute properties

    x = zeros(R, size(mb))
    x[mask] .= xvec
    return mb.*(R(1).+x)

end
function abs2contr(m::Array{R, 2}, mask::BitArray{2}, mb::Array{R, 2})

	return ((m.-mb)./mb)[mask]

end


function gradprec_contr2abs(grad::Array{R, 2}, mask::BitArray{2}, mb::Array{R, 2})
# Postprocessing steps: - preconditioning by diagonal background model

    return mb[mask].*grad[mask]

end
function gradprec_contr2abs(grad::Array{R, 2}, mask::BitArray{2}, mb::Model)

    return gradprec_contr2abs(grad, mask, mb.m)

end
