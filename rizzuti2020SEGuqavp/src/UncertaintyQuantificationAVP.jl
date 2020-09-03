# Author: Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
# Date: August, 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Uncertainty quantification module for AVP

module UncertaintyQuantificationAVP

using LinearAlgebra, FFTW, DSP
using Statistics, Random
using Flux, Flux.Optimise, InvertibleNetworks

# Utilities
include("./utils/datatypes.jl")
include("./utils/logobj.jl")
include("./utils/derivatives.jl")
include("./utils/density_utils.jl")
include("./utils/training_utils.jl")

# Modeling routines
include("./Modeling/mod_utils.jl")
include("./Modeling/modeling.jl")
include("./Modeling/wavelet.jl")
include("./Sampling/langevinSampling.jl")

# Invertible networks (1D)
import InvertibleNetworks: forward, inverse, backward, clear_grad!, get_params
include("./InvertibleNetworks1D/invertible_network_HINT1D.jl")
include("./InvertibleNetworks1D/invertible_network_Glow1D.jl")
include("./InvertibleNetworks1D/invertible_network_Hyperbolic1D.jl")
include("./InvertibleNetworks1D/invertible_network_utils.jl")

end
