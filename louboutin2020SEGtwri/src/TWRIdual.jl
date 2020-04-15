################################################################################
#
# TWRI module
#
################################################################################



module TWRIdual

using LinearAlgebra
using JUDI.TimeModeling
using PyCall
using Distributed
using FFTW, DSP
using SetIntersectionProjection
using Caching

const R = Float32; export R
const C = ComplexF32; export C
const SPACE_ORDER = 8
const NB = 40

# TWRI julia sources
include("TWRI_jl/utils.jl")
include("TWRI_jl/gendata.jl")
include("TWRI_jl/data_filter.jl")
include("TWRI_jl/FWIFun.jl")
include("TWRI_jl/TWRIdualFun.jl")


end
