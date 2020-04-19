using JLD2, PyPlot, JUDI.TimeModeling

PyPlot.rc("font", family="serif")
PyPlot.rc("xtick", labelsize=20) 
PyPlot.rc("ytick", labelsize=20)

@load "./data/bg_tti_g.jld"

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
imshow(model_true.theta', vmin=-.5, vmax=.5, cmap="jet", extent=extent, aspect="auto")
xlabel("X (m)", fontsize=20)
ax1 = gca()
ax1.set_yticks([])
title("θ", fontsize=20)
colorbar(ax=ax1)

savefig("./data/bg_model.png", bbox_inches="tight")

# Plot model perturbations
# True water layer
@load "./data/bg_tti_g.jld"
plt.figure(figsize=(30, 30))
subplot(221)
imshow(model0_tti.m', vmin=1/4.5^2, vmax=.44, cmap="jet", extent=extent, aspect="auto")
ylabel("Depth (m)", fontsize=20)
ax1 = gca()
ax1.set_xticks([])
title("m0", fontsize=20)
ax1 = gca()
colorbar(ax=ax1)
subplot(223)
imshow(model0_tti.m' - model_true.m', vmin=-.05, vmax=.05, cmap="jet", extent=extent, aspect="auto")
xlabel("X (m)", fontsize=20)
ylabel("Depth (m)", fontsize=20)
ax1 = gca()
title("δ m", fontsize=20)
colorbar(ax=ax1)
# Wrong water layer
@load "./data/bg_tti_g_w.jld"
subplot(222)
imshow(model0_tti.m', vmin=1/4.5^2, vmax=.44, cmap="jet", extent=extent, aspect="auto")
ax1 = gca()
ax1.set_yticks([])
ax1.set_xticks([])
title("m0", fontsize=20)
colorbar(ax=ax1)
subplot(224)
imshow(model0_tti.m' - model_true.m', vmin=-.05, vmax=.05, cmap="jet", extent=extent, aspect="auto")
xlabel("X (m)", fontsize=20)
ax1 = gca()
ax1.set_yticks([])
title("δ m", fontsize=20)
colorbar(ax=ax1)

savefig("./data/bg_dm.png", bbox_inches="tight")



### gradients with true water
@load "./data/bg_tti_g.jld"

# true tti params
figure(figsize=(30, 30))
subplot(321)
imshow(g_wri_tti', vmin=-.1, vmax=.1, cmap="jet", extent=extent, aspect="auto")
ylabel("Depth (m)", fontsize=20)
ax1 = gca()
ax1.set_xticks([])
title("WRI true anisotropy", fontsize=20)
colorbar(ax=ax1)
subplot(322)
imshow(g_fwi_tti', vmin=-.1, vmax=.1, cmap="jet", extent=extent, aspect="auto")
ax1 = gca()
ax1.set_xticks([])
ax1.set_yticks([])
title("FWI true anisotropy", fontsize=20)
colorbar(ax=ax1)

# Error in tti params
@load "./data/bg_tti_g_w.jld"
subplot(323)
imshow(g_wri_tti_w', vmin=-.05, vmax=.05, cmap="jet", extent=extent, aspect="auto")
ylabel("Depth (m)", fontsize=20)
ax1 = gca()
ax1.set_xticks([])
title("WRI inaccurate anisotropy", fontsize=20)
colorbar(ax=ax1)
subplot(324)
imshow(g_fwi_tti_w', vmin=-.1, vmax=.1, cmap="jet", extent=extent, aspect="auto")
ax1 = gca()
ax1.set_xticks([])
ax1.set_yticks([])
title("FWI inaccurate anisotropy", fontsize=20)
colorbar(ax=ax1)

# Acoustic
@load "./data/bg_acou_g.jld"
subplot(325)
imshow(g_wri_a', vmin=-.01, vmax=.01, cmap="jet", extent=extent, aspect="auto")
xlabel("X (m)", fontsize=20)
ylabel("Depth (m)", fontsize=20)
ax1 = gca()
title("WRI acoustic", fontsize=20)
colorbar(ax=ax1)
subplot(326)
imshow(g_fwi_a', vmin=-.1, vmax=.1, cmap="jet", extent=extent, aspect="auto")
xlabel("X (m)", fontsize=20)
ax1 = gca()
ax1.set_yticks([])
title("FWI acoustic", fontsize=20)
colorbar(ax=ax1)

savefig("./data/bg_true_water.png", bbox_inches="tight")


# Wrong water layer
@load "./data/bg_tti_g_ww.jld"

# true tti params
figure(figsize=(30, 30))
subplot(321)
imshow(g_wri_tti_ww', vmin=-.1, vmax=.1, cmap="jet", extent=extent, aspect="auto")
ylabel("Depth (m)", fontsize=20)
ax1 = gca()
ax1.set_xticks([])
title("WRI true anisotropy", fontsize=20)
colorbar(ax=ax1)
subplot(322)
imshow(g_fwi_tti_ww', vmin=-.1, vmax=.1, cmap="jet", extent=extent, aspect="auto")
ax1 = gca()
ax1.set_xticks([])
ax1.set_yticks([])
title("FWI true anisotropy", fontsize=20)
colorbar(ax=ax1)

# Error in tti params
@load "./data/bg_tti_g_ww_w.jld"
subplot(323)
imshow(g_wri_tti_ww_w', vmin=-.05, vmax=.05, cmap="jet", extent=extent, aspect="auto")
ylabel("Depth (m)", fontsize=20)
ax1 = gca()
ax1.set_xticks([])
title("WRI inaccurate anisotropy", fontsize=20)
colorbar(ax=ax1)
subplot(324)
imshow(g_fwi_tti_ww_w', vmin=-.1, vmax=.1, cmap="jet", extent=extent, aspect="auto")
ax1 = gca()
ax1.set_xticks([])
ax1.set_yticks([])
title("FWI inaccurate anisotropy", fontsize=20)
colorbar(ax=ax1)

# Acoustic
@load "./data/bg_acou_g_ww.jld"
subplot(325)
imshow(g_wri_a_ww', vmin=-.01, vmax=.01, cmap="jet", extent=extent, aspect="auto")
xlabel("X (m)", fontsize=20)
ylabel("Depth (m)", fontsize=20)
ax1 = gca()
title("WRI acoustic", fontsize=20)
colorbar(ax=ax1)
subplot(326)
imshow(g_fwi_a_ww', vmin=-.1, vmax=.1, cmap="jet", extent=extent, aspect="auto")
xlabel("X (m)", fontsize=20)
ax1 = gca()
ax1.set_yticks([])
title("FWI acoustic", fontsize=20)
colorbar(ax=ax1)

savefig("./data/bg_wrong_water.png", bbox_inches="tight")
