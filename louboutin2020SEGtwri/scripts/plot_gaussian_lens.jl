using JLD2, PyPlot, JUDI.TimeModeling

PyPlot.rc("font", family="serif")
PyPlot.rc("xtick", labelsize=20) 
PyPlot.rc("ytick", labelsize=20)

@load "./data/tti_g.jld"

n = model_true.n
d = model_true.d
extent = [0, (n[1] -1) * d[1],  (n[2] -2) * d[2], 0]


### Plot model
figure(figsize=(30, 30))
subplot(221)
imshow(model_true.m', vmin=1/4.5^2, vmax=.44, cmap="jet", extent=extent, aspect="auto")
ax1 = gca()
ax1.set_xticks([])
ylabel("Depth (m)", fontsize=20)
title("m", fontsize=20)
colorbar(ax=ax1)

subplot(222)
imshow(model_true.epsilon', vmin=0.0, vmax=.3, cmap="jet", extent=extent, aspect="auto")
ax1 = gca()
ax1.set_xticks([])
ax1.set_yticks([])
title("ε", fontsize=20)
colorbar(ax=ax1)

subplot(223)
imshow(model_true.delta', vmin=0.0, vmax=.3, cmap="jet", extent=extent, aspect="auto")
ax1 = gca()
xlabel("X (m)", fontsize=20)
ylabel("Depth (m)", fontsize=20)
title("δ", fontsize=20)
colorbar(ax=ax1)

subplot(224)
imshow(model_true.theta', vmin=-.25, vmax=.25, cmap="jet", extent=extent, aspect="auto")
xlabel("X (m)", fontsize=20)
ax1 = gca()
ax1.set_yticks([])
title("θ", fontsize=20)
colorbar(ax=ax1)

savefig("./data/gl_model.png", bbox_inches="tight")


### Plot inital model and perturbationgradients
plt.figure(figsize=(30, 30))
subplot(121)
imshow(model0_tti.m', vmin=1/4.5^2, vmax=.44, cmap="jet", extent=extent, aspect="auto")
xlabel("X (m)", fontsize=20)
ylabel("Depth (m)", fontsize=20)
ax1 = gca()
title("m0", fontsize=20)
ax1 = gca()
colorbar(ax=ax1)

subplot(122)
imshow(model0.m' - model_true.m', vmin=-.05, vmax=.05, cmap="jet", extent=extent, aspect="auto")
xlabel("X (m)", fontsize=20)
ax1 = gca()
ax1.set_yticks([])
title("δ m", fontsize=20)
colorbar(ax=ax1)

savefig("./data/gl_dm.png", bbox_inches="tight")


### Plot gradients

# True tti params
@load "tti_g.jld"
figure(figsize=(30, 30))
subplot(321)
imshow(g_wri_tti', vmin=-1, vmax=1, cmap="jet", extent=extent, aspect="auto")
ylabel("Depth (m)", fontsize=20)
ax1 = gca()
ax1.set_xticks([])
# ax1.set_yticks([])
title("WRI true anisotropy", fontsize=20)
colorbar(ax=ax1)
subplot(322)
imshow(g_fwi_tti', vmin=-.5, vmax=.5, cmap="jet", extent=extent, aspect="auto")
ax1 = gca()
ax1.set_xticks([])
ax1.set_yticks([])
title("FWI true anisotropy", fontsize=20)
colorbar(ax=ax1)

# Error in tti params
@load "tti_g_w.jld"
subplot(323)
imshow(g_wri_tti_w', vmin=-1, vmax=1, cmap="jet", extent=extent, aspect="auto")
ylabel("Depth (m)", fontsize=20)
ax1 = gca()
ax1.set_xticks([])
title("WRI inaccurate anisotropy", fontsize=20)
colorbar(ax=ax1)
subplot(324)
imshow(g_fwi_tti_w', vmin=-.5, vmax=.5, cmap="jet", extent=extent, aspect="auto")
ax1 = gca()
ax1.set_xticks([])
ax1.set_yticks([])
title("FWI inaccurate anisotropy", fontsize=20)
colorbar(ax=ax1)

# Acosutic WE gradient
@load "acou_g.jld"
subplot(325)
imshow(g_wri_a', vmin=-1, vmax=1, cmap="jet", extent=extent, aspect="auto")
xlabel("X (m)", fontsize=20)
ylabel("Depth (m)", fontsize=20)
ax1 = gca()
title("WRI acoustic", fontsize=20)
colorbar(ax=ax1)
subplot(326)
imshow(g_fwi_a', vmin=-.5, vmax=.5, cmap="jet", extent=extent, aspect="auto")
xlabel("X (m)", fontsize=20)
ax1 = gca()
ax1.set_yticks([])
title("FWI acoustic", fontsize=20)
colorbar(ax=ax1)

savefig("./data/gl_wrong.png", bbox_inches="tight")
