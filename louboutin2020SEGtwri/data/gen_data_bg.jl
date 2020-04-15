################################################################################
#
# Generate synthetic data for the BG Compass model
#
################################################################################



### Module loading


using Distributed, JLD2, PyPlot, SegyIO, Images, PyCall, JUDI.TimeModeling
@everywhere push!(LOAD_PATH, string(pwd(), "/src/")); using TWRIdual


### Load true model


# Original size
d = (12.5f0, 12.5f0)
o = (0f0, 0f0)

@load "bg_tti.jld"

# Model

n = size(m_true)
model_true = Model_TTI(n, d, o, m_true; epsilon=epsilon, delta=delta, theta=theta)

### Acquisition geometrye
dt = 4f0
T = 2500f0 # total recording time [ms]

# Source wavelet
freq_peak = 0.01f0
wavelet = ricker_wavelet(T, dt, freq_peak)

# Sources
nsrc = 51
x_src = convertToCell(range(0f0, stop = (size(m_true, 1)-1)*d[1], length = nsrc))
y_src = convertToCell(range(0f0, stop = 0f0, length = nsrc))
z_src = convertToCell(range(0f0, stop = 0f0, length = nsrc))

# Receivers
nrcv = size(m_true, 1)
x_rcv = range(0f0, stop = (size(m_true, 1)-1)*d[1], length = nrcv)
y_rcv = 0f0
z_rcv = range(12.5f0, stop = 12.5f0, length = nrcv)

# Geometry structures
src_geom = Geometry(x_src, y_src, z_src; dt = dt, t = T)
rcv_geom = Geometry(x_rcv, y_rcv, z_rcv; dt = dt, t = T, nsrc = nsrc)

# Source function
fsrc = judiVector(src_geom, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(src_geom, rcv_geom, model_true)
info = Info(prod(n), nsrc, ntComp)

# Setup operators
F = judiModeling(info,model_true, src_geom,rcv_geom)
dat = F*fsrc

### Saving data
@save string("./data/BGCompass_data_tti.jld") model_true fsrc dat
