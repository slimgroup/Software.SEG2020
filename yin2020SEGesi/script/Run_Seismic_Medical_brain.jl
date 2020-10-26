# Example for Variable Projection:
# Author: Ziyi Yin, Rafael Orozco, Philipp Witte, Mathias Louboutin, Gabrio Rizzuti, Felix J. Herrmann
# Date: April, 2020
#
using DrWatson
@quickactivate "ExtSrcImg"

include("gen_geometry.jl")
include("modeling_extended_source_spg_raf_function.jl")


using PyPlot, FFTW, Images
using JUDI.TimeModeling, JOLI
using LinearAlgebra, PyPlot, Random, Statistics
using IterativeSolvers
using JLD
using MAT
using Random

function weight(n, d, xs, zs, nsrc,mode)
	if mode == "medical"
		zs_ave = zeros(1,n[2],nsrc); #hacky way
		zs_ave .= mean(zs)

		xs_ave = zeros(1,n[2],nsrc); #hacky way
		xs_ave .= mean(zs)

		x = zeros(n[1],1,nsrc);
	    x .= reshape((collect(1:n[1]).-1)*d[1], :, 1)
		z = zeros(1,n[2],nsrc);
	    z .= reshape((collect(1:n[2]).-1)*d[2], 1, :)

		W = zeros(Float32,n[1],n[2],nsrc)
		#W .= sqrt.((z.-zs_ave).^2f0)
		 W .= sqrt.((x.-zs_ave).^2f0.+(z.-zs_ave).^2f0)
		

		# x = zeros(n[1],1,nsrc);
	 #    x .= reshape((collect(1:n[1]).-1)*d[1], :, 1)
		# z = zeros(1,n[2],nsrc);
	 #    z .= reshape((collect(1:n[2]).-1)*d[2], 1, :)
		# W = zeros(Float32,n[1],n[2],nsrc)
		# W .= sqrt.((x.-xs).^2f0.+(z.-zs).^2f0)
		
	elseif mode == "seismic"
		x = zeros(n[1],1,nsrc);
	    x .= reshape((collect(1:n[1]).-1)*d[1], :, 1)
		z = zeros(1,n[2],nsrc);
	    z .= reshape((collect(1:n[2]).-1)*d[2], 1, :)
		W = zeros(Float32,n[1],n[2],nsrc)
		W .= sqrt.((x.-xs).^2f0.+(z.-zs).^2f0)
		#W = sqrt.(r.^2f0.+δ^2f0)./δ
	end
	
    return reshape(W,n[1],n[2],1,nsrc)
end

function CG(d_obs, q0, m0, W; maxiter=5,λ=1f2)
	# use CG to solve damped least square normal equations
	r1 = d_obs-𝒢(q0, m0);
	r = 𝒢_T(r1,m0)-λ*W.^2f0.*q0;
	p = r;
	r_vec = zeros(maxiter+1);
	println("residual: $(norm(r1)) before updating q")
	for i = 1:maxiter
		
		if norm(r1)<1f-3
			break
		end
		
		temp = sum(sum(r.*r,dims=1),dims=2);
		temp1 = 𝒢(p, m0);
		temp2 = 𝒢_T(temp1,m0)+λ*W.^2f0.*p;
		alpha = temp./sum(sum(p.*temp2,dims=1),dims=2);
		q0 = q0 + alpha.* p;
		r1 = r1 - alpha.* temp1;
		r = r - alpha .* temp2;
		beta = sum(sum(r.*r,dims=1),dims=2)./temp;
		p = r + beta.*p;
		r_vec[i+1]=norm(r1)
		println("residual: $(norm(r1)) after updating q")
		
	end
	return q0, r1, r_vec
end

# Objective function: estimate source and compute residual + function value
function objective(d_obs, q0, m0, W; λ=1f2)

	# Data residual and source residual
  	r_data   = 𝒢(q0, m0) - d_obs
	r_source = q0

	# Function value
	fval = .5f0 *sum(r_data.^2) + 0.5f0*λ*sum((W.*r_source).^2)
	
	print("Function value: ", fval, "\n")
	return fval
end

function GenSimSourceMulti(xsrc_index, zsrc_index, nsrc, n)
	weights = zeros(Float32, n[1], n[2], 1, nsrc)

	num_simultaneous_sources = size(xsrc_index)[1]
	#check if Simultaneous
	if num_simultaneous_sources > 1 && nsrc == 1
		#for Simultaneous we need to put all the weights in the same slice. 
		for j=1:num_simultaneous_sources
			
			
        	weights[xsrc_index[j], zsrc_index[j], 1, 1] = 1f0
    	end
	else
		for j=1:nsrc
        	weights[xsrc_index[j], zsrc_index[j], 1, j] = 1f0
    	end
	end
    
    return weights
end

function GenSimData(q₀; model=model)   # pass q₀ directly instead of z
    # Simultaneous observed data
    d_obs = F*vec(q₀)
    d_obs = reshape(d_obs, recGeometry.nt[1], nrec, 1, nsrc)  # reshape into tensor
    return d_obs
end

function objective_function_spg(x)
	#cFact = Float32(1 ./model.d[1]^2); #scale model up to make gradient have a normal magnitude

	r = d_obs-𝒢(x, m0);
  	f = .5f0*norm(r)^2;
  	g = -vec(𝒢_T(r,m0));

  	return f, (g/norm(g, Inf));
end


############################################ Models ###############################################

M = load("../data/brain.jld")
m = M["m"]
m = m./1f6
N = load("../data/mute_sp_phantom.jld")
mute = N["mute"]
m0 = zeros(size(m))
m0 .= m[1,1]
#m0 = imfilter(deepcopy(m),Kernel.gaussian(5))
n = size(m)
dx = 50f0 #from hauptmann
dz = 50f0   # in micro meters (μm)
#dx = 687f0 #from hauptmann
#dz = 687f0   # in micro meters (μm)


d            = (dx,dz);
o            = (0.,0.);
#the only difference between the models is the slowness

model = Model(n, d, o, m,nb=120)
model0 = Model(n, d, o, m0,nb=120)

extentx = n[1]*d[1] / 1000 #in mm
extentz = n[2]*d[2] / 1000 #in mm

# receiver sampling and recording time
dtR = 0.02f0  # micro seconds (μs)
timeR = 25
#dtR = 2f0  # micro seconds (μs)
#timeR = 1500f0

timeS = timeR
dtS = dtR

#Mode = "seismic"
mode = "medical"

if mode == "seismic"
	# Set up receiver structure
	xrec,yrec,zrec,nrec,nsrc =GenRecGeometrySeismic();
	recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

	# Set up source structure
	xsrc_index,zsrc_index = GenSrcGeometrySeismic(nsrc);

	#generate tensor with sources
	q       = GenSimSourceMulti(xsrc_index, zsrc_index, nsrc, model.n);
	
	#get muting region
	mute_mask = GenMuteSeismic()

elseif mode == "medical"

	xrec, yrec, zrec, nrec, nsrc, xsrc_index, zsrc_index, nsrc_simult = Gen_Exp_Medical_Skull_Phantom_Source()
	
	#generate tensor with sources
	q           = GenSimSourceMulti(xsrc_index, zsrc_index, nsrc, model.n);

	#get muting region
	mute_mask = GenMuteMedical_brain()
else
	error("Please choose a correct mode.")
end

recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# Setup wavelet
#f0 = 0.0035  # dominant frequency in [kHz]
f0 = 5  # dominant frequency in [MHz]
#f0 = 0.0030  # dominant frequency in [kHz]
wavelet = ricker_wavelet(timeS,dtS,f0)

######################################### Set up JUDI operators ###################################

# Set up info structure for linear operators
ntComp = get_computational_nt(recGeometry, model)
info   = Info(prod(model0.n), nsrc, ntComp)

# Write shots as array
opt = Options(return_array = true, subsampling_factor=10, isic=true)
#opt = Options(return_array = true, isic=true)

# Modeling operators
Pr  = judiProjection(info, recGeometry)
F0  = judiModeling(info, model0; options=opt)	# modeling operator true model
F  = judiModeling(info, model; options=opt)	# modeling operator initial model
Pw = judiLRWF(info, wavelet)

# Combined operators
F = Pr*F*adjoint(Pw)
F0 = Pr*F0*adjoint(Pw)

# Extended modeling CNN layers
𝒢 = ExtendedQForward(F)
𝒢_T = ExtendedQAdjoint(F)

######################################### Run Forward simulations ###################################
# Generate shots
d_obs   = GenSimData(q); # Data in true model

######################################### Generate 3 gradient images ###################################

#run cg to invert for estimated source
q_iter = 25

q_init = zeros(Float32, n[1], n[2]);
q_inv = spg_tv_norm(d_obs, objective_function_spg, sum(q,dims=4)[:,:,1,1], q_init, model0; maxiter=q_iter, tv_norm=0.5)

#get jacobian using estimated source and use it to get gradient
J = judiJacobian(F0, q_inv)
grad_extend=reshape(adjoint(J)* vec(d_obs-𝒢(q_inv,m0)),n[1],n[2])

figure(2);imshow(grad_extend'/norm(grad_extend',Inf),cmap="gray",extent=(0,extentx,extentz,0),vmin=-0.05,vmax=0.05);title("Extended Source Imaging - Inversion");
xlabel("Lateral position [mm]");ylabel("Depth [mm]");
cb = colorbar(fraction=0.038, pad=0.04);
fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)

#### 3.) Extended source gradient - adjoint scheme
q_adj = 𝒢_T(d_obs,m0)

print(vars)
######################################### Visualize results   ###################################


#show models used

v = reshape(sqrt.(1f0./(m.*1f6)),n[1], n[2]);
using PyCall
@pyimport numpy.ma as ma
v1 = pycall(ma.masked_less, Any, v', 2)
v2 = pycall(ma.masked_greater, Any, v', 2)

PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
fig,ax = PyPlot.subplots()
p2 = ax.imshow(v2,extent=(0,extentx,extentz,0),interpolation="nearest",cmap="winter",vmin=1.5,vmax=1.55);
cb2 = colorbar(p2,fraction=0.037, pad=0.08);
p1 = ax.imshow(v1,extent=(0,extentx,extentz,0),interpolation="nearest",cmap="Reds",vmin=2.45,vmax=2.55)
cb1 = colorbar(p1,fraction=0.043, pad=0.08);
PyPlot.scatter(xrec/1000,zrec/1000,marker=".",color="yellow",label="receivers",s=2);
PyPlot.scatter(xsrc_index*d[1]/1000,zsrc_index*d[2]/1000,label="sources",color="white",s=1);
legend(loc=1,fontsize=7);
xlabel("Lateral position [mm]", fontsize=8);ylabel("Horizontal position [mm]", fontsize=8);
fig = PyPlot.gcf(); fig.set_size_inches(6, 4.5)
savefig("phantom_setup.png")

if mode == "seismic"	# one single extended source in the middle

	q_inv_stack=sum(q_inv,dims=4)[:,:,1,1];
	figure(8);imshow(q_inv_stack',vmin=-2f-3,vmax=2f-3,extent=(0,extentx,extentz,0));title("Extended Source from Inversion");
	xlabel("Lateral position [samples]");ylabel("Depth [samples]");
	colorbar(fraction=0.028, pad=0.04);
	fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)

	q_adj_stack=sum(q_adj,dims=4)[:,:,1,1];
	figure(9);imshow(q_adj_stack',vmin=-40000,vmax=40000,extent=(0,extentx,extentz,0));title("Extended Source from Adjoint");
	xlabel("Lateral position [samples]");ylabel("Depth [samples]");
	colorbar(fraction=0.028, pad=0.04);
	fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)

	if nsrc != 1
		mid_loc = Int(nsrc/2)
		q_inv_single=q_inv[:,:,1,mid_loc];
		figure(18);imshow(q_inv_single',vmin=-2f-3,vmax=2f-3,extent=(0,extentx,extentz,0));title("Extended Source from Inversion");
		xlabel("Lateral position [samples]");ylabel("Depth [samples]");
		colorbar(fraction=0.028, pad=0.04);
		fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)

		q_adj_single=q_adj[:,:,1,mid_loc];
		figure(19);imshow(q_adj_single',vmin=-40000,vmax=40000,extent=(0,extentx,extentz,0));title("Extended Source from Adjoint");
		xlabel("Lateral position [samples]");ylabel("Depth [samples]");
		colorbar(fraction=0.028, pad=0.04);
		fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)
	end

elseif mode == "medical"	
	PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
	figure(); imshow(reshape(q_adj,n[1], n[2])'.*mute',extent=(0,extentx,extentz,0),vmin=-0.8*norm(q_adj,Inf),vmax=0.8*norm(q_adj,Inf));
	xlabel("Lateral position [mm]", fontsize=8);ylabel("Horizontal position [mm]", fontsize=8);
	fig = PyPlot.gcf(); fig.set_size_inches(6, 4.5)
	savefig("phantom_adj_rand.png")

	PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
	figure(); imshow(reshape(q_inv,n[1], n[2])'.*mute',extent=(0,extentx,extentz,0),vmin=-0.8*norm(q_inv,Inf),vmax=0.8*norm(q_inv,Inf));
	xlabel("Lateral position [mm]", fontsize=8);ylabel("Horizontal position [mm]", fontsize=8);
	fig = PyPlot.gcf(); fig.set_size_inches(6, 4.5)
	savefig("phantom_inv_water.png")
	
	PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
	figure(); imshow(reshape(q,n[1], n[2])'.*mute',extent=(0,extentx,extentz,0));
	cb = colorbar(fraction=0.037, pad=0.04);
	title("True Source", fontsize=8);
	xlabel("Lateral position [mm]", fontsize=8);ylabel("Horizontal position [mm]", fontsize=8);
	fig = PyPlot.gcf(); fig.set_size_inches(6.66, 5)
	savefig("phantom_true.png", bbox_inches="tight")
end


######################################### Save results   ########################################
#@save "tv_brain_water.jld" q q_inv q_adj m m0 d n xrec yrec zrec xsrc_index zsrc_index grad_extend mute d_obs
