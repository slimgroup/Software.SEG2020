################################################################################
#
# RGradient comparison for the BG Compass model
#
################################################################################
### Module loading
using LinearAlgebra, JLD2, PyPlot, Images
using JUDI.TimeModeling
push!(LOAD_PATH, string(pwd(), "/src/"));
using TWRIdual

### Load synthetic data
@load "./data/BGCompass/BGCompass_data_tti.jld"

# Filter data
freq = 0.003f0
df = .001f0
dat2 = low_filter(dat, dat.geometry.dt[1]; fmin=1f3*(freq-df), fmax=1f3*(freq+df))
fsrc2 = low_filter(fsrc, fsrc.geometry.dt[1]; fmin=1f3*(freq-df), fmax=1f3*(freq+df))


### Background model
idx_w = 0
var = 20
n = model_true.n
d = model_true.d



function compute_gradients(model0, fsrc, dat, idx_w)
    m0 = model0
    # Time sampling
    dt_comp = get_dt(model0)
    nt_comp = get_computational_nt(fsrc.geometry, dat.geometry, model0)

    # Pre- and post-conditioning
    mask = BitArray(undef, model0.n)
    mask .= false
    mask[:, idx_w+1:end] .= true
    preproc(x) = contr2abs(x, mask, m0)
    postproc(g) = gradprec_contr2abs(g, mask, m0)

    # [FWI]
    inv_name = "FWI"
    fun!(F, G, x) = objFWI!(F, G, preproc(x), model0, fsrc, dat; gradprec_fun=postproc)

    # [TWRIdual]
    inv_name = "TWRIdual"

    ε0 = 0.0f0
    ε = Array{Float32, 1}(undef, fsrc.nsrc)
    ε .= ε0
    grad_corr = false
    objfact = 1f0
    v_bg = sqrt(1/m0[1])
    freq_peak = 0.003f0
    δ = 1f0*R(sqrt(2)/2)*v_bg/freq_peak
    weight_fun_pars = ("srcfocus", δ)
    fun!(F, G, x, weight_fun_pars, objfact) = objTWRIdual!(F, G, preproc(x), model0, fsrc, dat, ε;
                                                        objfact = objfact, comp_alpha = true, gradprec_fun=postproc,
                                                        grad_corr = grad_corr, weight_fun_pars = weight_fun_pars)

    fun_wri!(F, G, x) = fun!(F, G, x, weight_fun_pars, objfact)
    ### Computing gradients
    x0 = zeros(R, length(findall(mask .== true)))
    m_inv = preproc(x0)
    G_FWI = zeros(R, n)
    G = zeros(R, size(mask))
    fun!(true, G, x0)
    G_FWI[mask] .= G
    G_FWI = G_FWI/norm(G_FWI, Inf)

    ### Computing gradients
    x02 = zeros(R, length(findall(mask .== true)))
    m_inv2 = preproc(x0)
    G_WRI = zeros(R, n)
    G2 = zeros(R, size(mask))
    fun_wri!(true, G2, x02)
    G_WRI[mask] .= G2
    G_WRI = G_WRI/norm(G_WRI, Inf)


    return G_FWI, G_WRI
end

### True water
idx_w = 17
# True Thomsen parameters model
model0_tti = deepcopy(model_true)
model0_tti.m .= R.(imfilter(model0.m[:, idx_w+1:end], Kernel.gaussian(var)))

g_fwi_tti, g_wri_tti = compute_gradients(model0_tti, fsrc2, dat2, idx_w)
@save "./data/bg_tti_g.jld" g_fwi_tti g_wri_tti model0_tti model_true

# Erronous Thomsen parameters model
model0_tti_err = deepcopy(model_true)

model0_tti_err.m .= model0_tti.m
model0_tti_err.epsilon[:, idx_w+1:end] = R.(imfilter(model0_tti_err.epsilon[:, idx_w+1:end], Kernel.gaussian(var)))
model0_tti_err.delta[:, idx_w+1:end] = R.(imfilter(model0_tti_err.delta[:, idx_w+1:end], Kernel.gaussian(var)))
model0_tti_err.theta[:, idx_w+1:end] = R.(imfilter(model0_tti_err.theta[:, idx_w+1:end], Kernel.gaussian(var)))


g_fwi_tti_w, g_wri_tti_w = compute_gradients(model0_tti_err, fsrc2, dat2, idx_w)
@save "./data/bg_tti_g_w.jld" g_fwi_tti_w g_wri_tti_w model0_tti_err model_true

# Acoustic model
model0_acou = Model(n, d, o, model0_tti.m)

g_fwi_a, g_wri_a = compute_gradients(model0_acou, fsrc2, dat2, idx_w)
@save "./data/bg_acou_g.jld" g_fwi_a g_wri_a model0_acou model_true


### Wrong water
idx_w = 0
# True Thomsen parameters model
model0_tti = deepcopy(model_true)
model0_tti.m .= R.(imfilter(model0.m[:, idx_w+1:end], Kernel.gaussian(var)))

g_fwi_tti, g_wri_tti = compute_gradients(model0_tti, fsrc2, dat2, idx_w)
@save "./data/bg_tti_g_w.jld" g_fwi_tti g_wri_tti model0_tti model_true

# Erronous Thomsen parameters model
model0_tti_err = deepcopy(model_true)

model0_tti_err.m .= model0_tti.m
model0_tti_err.epsilon[:, idx_w+1:end] = R.(imfilter(model0_tti_err.epsilon[:, idx_w+1:end], Kernel.gaussian(var)))
model0_tti_err.delta[:, idx_w+1:end] = R.(imfilter(model0_tti_err.delta[:, idx_w+1:end], Kernel.gaussian(var)))
model0_tti_err.theta[:, idx_w+1:end] = R.(imfilter(model0_tti_err.theta[:, idx_w+1:end], Kernel.gaussian(var)))


g_fwi_tti_w, g_wri_tti_w = compute_gradients(model0_tti_err, fsrc2, dat2, idx_w)
@save "./data/bg_tti_g_w_w.jld" g_fwi_tti_w g_wri_tti_w model0_tti_err model_true

# Acoustic model
model0_acou = Model(n, d, o, model0_tti.m)

g_fwi_a, g_wri_a = compute_gradients(model0_acou, fsrc2, dat2, idx_w)
@save "./data/bg_acou_g_w.jld" g_fwi_a g_wri_a model0_acou model_true
