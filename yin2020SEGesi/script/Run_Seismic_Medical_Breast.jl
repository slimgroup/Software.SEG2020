# Author: Ziyi Yin, Rafael Orozco, Philipp Witte, Mathias Louboutin, Gabrio Rizzuti, Felix J. Herrmann
# Date: April, 2020

using DrWatson
@quickactivate "ExtSrcImg"

include("gen_geometry.jl")

using PyPlot, FFTW, Images
using JUDI, JOLI, JUDI4Flux
using LinearAlgebra, PyPlot, Random, Statistics
using IterativeSolvers
using JLD
using MAT
using Distributions

function weight(n, d, xs, zs, nsrc,mode)
	
	x = zeros(n[1],1,nsrc);
    x .= reshape((collect(1:n[1]).-1)*d[1], :, 1)
	z = zeros(1,n[2],nsrc);
    z .= reshape((collect(1:n[2]).-1)*d[2], 1, :)
	W = zeros(Float32,n[1],n[2],nsrc)
	W .= sqrt.((x.-xs).^2f0.+(z.-zs).^2f0)
	
    return reshape(W,n[1],n[2],1,nsrc)
end

function CG(d_obs, q0, m0, W; maxiter=5,λ=1f2)
	# use CG to solve damped least square normal equations
	r1 = d_obs-𝒢(q0, m0);
	r = 𝒢_T(r1,m0)-λ*W.^2f0.*q0;
	p = r;
	r_vec = zeros(maxiter+1);
	r_vec[1] = norm(r1)
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
  	r_data   = d_obs-𝒢(q0, m0)
	r_source = q0

	# Function value
	fval = .5f0 *sum(r_data[:,:,1,1].^2) + 0.5f0*λ*sum((W.*r_source)[:,:,1,1].^2)
	
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
        	weights[xsrc_index[j], zsrc_index[j], 1, j] = 1#
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

function make_calcifications(x=100,z=100,spread=80, n=20)
	#make n calcifications centered around x,y with a certain spread
	normal_two_dim = MvNormal(vec([x z]), Diagonal(spread*ones(2)))
	norm_coords =convert(Array{Int64,2},round.(rand(normal_two_dim, n)))
	norm_coords_x = norm_coords[1,:]
	norm_coords_z = norm_coords[2,:]
	for i in 1:n
		#use random size and location to place "microcalcification" of high velocity
		box_size_x = rand(0:2)
		box_size_z = rand(0:2)
		v[x+norm_coords_x[i]:x+norm_coords_x[i]+box_size_x, z+norm_coords_z[i]:z+norm_coords_z[i]+box_size_z] .= 3432.03 #in micrometers/microsecond
	end
end


############################################ Models ###############################################
M = load("../data/breast_ring.jld") #load breast velocity model 

v = M["v"][1:601,1:601] * 1000 #bring v to micrometers / microseconds

size_x = size(v)[1]
size_z = size(v)[2]
n = (size_x,size_z)

# define smoothed velocity model for inversion NOTE: this is before we add micro calcifications
v0 = imfilter(deepcopy(v), Kernel.gaussian(10));

#add calcifications to forward model 
Random.seed!(3) #set seed to get same microcalcifications. 
make_calcifications(160,130,80,20)

#setup models 
dx = 25 #in micrometers
dz = 25 #in micrometers
m  = 1f0 ./v.^2; 
d  = (dx,dz);
o  = (0.,0.);

#the only difference between the forward and inversion model is the velocity model 
m0       = 1f0 ./v0.^2; #s^2/m^2

model  = Model(n, d, o,  m, nb=80) #extra boundary layer size (nb) to avoid reflections artifacts
model0 = Model(n, d, o, m0, nb=80) #extra boundary layer size (nb) to avoid reflections artifacts

m = reshape(model.m, n[1], n[2], 1, 1)
m0 = reshape(model0.m, n[1], n[2], 1, 1)

##setup sensor sampling time values in microseconds
dtR = 0.00698    
timeR = 3.4912718*3#in microseconds.

#source will have same time as sensors
timeS = timeR
dtS = dtR

#mode = "seismic"
mode = "medical"
include("gen_geometry.jl")

if mode == "seismic"

	####----new experiment with seismic like multishots
	xrec, yrec, zrec, nrec, nsrc, xsrc_index, zsrc_index, nsrc_simult = Gen_Exp_Seismic_Calc_Breast_Blood()
	
	#generate tensor with sources
	q           = GenSimSourceMulti(xsrc_index, zsrc_index, nsrc, model.n);

elseif mode == "medical"


	xrec, yrec, zrec, nrec, nsrc, xsrc_index, zsrc_index, nsrc_simult = Gen_Exp_Medical_Calc_Breast_Single()
	
	#generate tensor with sources
	q           = GenSimSourceMulti(xsrc_index, zsrc_index, nsrc, model.n);

	#get muting region
	mute_mask = GenMuteMedical_breast();
else
	error("Please choose a correct mode.")
end

recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# Setup wavelet
f_dom = 3.25 #in [mHz]
wavelet = ricker_wavelet(timeS,dtS,f_dom)

######################################### Set up JUDI operators ###################################

# Set up info structure for linear operators
ntComp = get_computational_nt(recGeometry, model)
info   = Info(prod(model0.n), nsrc, ntComp)

# Write shots as array
opt = Options(return_array = true, subsampling_factor=5, isic=true)

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
d_obs   = GenSimData(q);  #Data in true model

######################################### Generate 3 gradient images ###################################

#### 1.) RTM
J1=judiJacobian(F0,q)
d_pred = 𝒢(q,m0)
res_rtm = d_obs-d_pred;
grad_rtm=reshape(adjoint(J1)* vec(res_rtm),n[1],n[2])


#### 2.) Extended source gradient - inversion scheme
W = 0 #no focusing
λ = 0 #no focusing

#run cg to invert for estimated source
q_iter = 5
q_inv = zeros(Float32, n[1], n[2], 1, nsrc); #initialize source guess at zero
q_inv, r_end, rvec = CG(d_obs, q_inv, m0, W; maxiter=q_iter,λ=λ)

#get jacobian using estimated source and use it to get gradient
J = judiJacobian(F0, q_inv)
grad_extend=reshape(adjoint(J)* vec(r_end),n[1],n[2])


######################################### Visualize results   ###################################
#get extents of model for plotting
pixel_size_x = 0.2 #in mm
pixel_size_z = 0.2 #in mm
extentx = n[1]*pixel_size_x  #in mm
extentz = n[2]*pixel_size_z  #in mm

#show stacked gradient image 

# Mute out
r = 30
x = range(-30, 30, length=size(grad_extend, 1))
z  = range(-30, 30, length=size(grad_extend, 1))

mask = zeros(size(grad_extend))

for i =1:size(grad_extend, 1)
	for j=1:size(grad_extend, 2)
		if (x[i]^2 + z[j]^2) < r^2
			mask[i, j] = 1
		end
	end
end

mask =imfilter(mask, Kernel.gaussian(5))
scale = .4 * 1e-1

fig=figure(figsize=(6.66,5));
imshow(-mask' .* grad_extend'/norm(grad_extend,Inf), cmap="Greys", vmin=-scale, vmax=scale , extent=(0, extentx, extentz,0))
title("Extended Source Image - Inversion - Stacked", fontsize=8)
xlabel("Lateral Position [mm]", fontsize=8)
ylabel("Horizontal Position [mm]", fontsize=8)
# colorbar(fraction=0.028, pad=0.04)

tight_layout()
savefig("breast_blood_stacked.png", dpi=300, format="png")
run(`mogrify -trim breast_blood_stacked.png`)


#Show solved sources and true sources
q_inv_stack=sum(q_inv,dims=4)[:,:,1,1];
figure(8);imshow(q_inv_stack',extent=(0,extentx,extentz,0));title("Extended Source from Inversion");
xlabel("Lateral position [mm]");ylabel("Depth [mm]");
colorbar(fraction=0.028, pad=0.04);
fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)

q_stack=sum(q,dims=4)[:,:,1,1];
figure(8);imshow(q_stack',extent=(0,extentx,extentz,0));title("True sources");
xlabel("Lateral position [mm]");ylabel("Depth [mm]");
colorbar(fraction=0.028, pad=0.04);
fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)



#show models used
figure(5);imshow(m0[:,:,1,1]',extent=(0,extentx,extentz,0));title("Background");
cb = colorbar(fraction=0.038, pad=0.04);
cb[:set_label]("medium velocity [m/s]")
xlabel("Lateral position [mm]");ylabel("Depth [mm]");
fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)

figure(7);imshow(m[:,:,1,1]',extent=(0,extentx,extentz,0));title("True");
xlabel("Lateral position [km]");ylabel("Depth [km]");colorbar(fraction=0.028, pad=0.04);
fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)


#Show Experiment Setup this is the figure in the paper
PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
fig=figure(figsize=(6.66,5));
imshow(reshape(v/1000, n[1], n[2])',vmin=1.47,vmax=1.70,extent=(0,extentx,extentz,0));
cb = colorbar(fraction=0.040, pad=0.04);
#cb[:set_label]("Medium Velocity [m/s]", size="large")
PyPlot.scatter((xrec/d[1])*pixel_size_x,(zrec/d[2])*pixel_size_z,marker=".",color="red",label="Receivers");
PyPlot.scatter(xsrc_index*pixel_size_x,zsrc_index*pixel_size_z,label="Sources",marker="x",color="white");
title("True Model - Experiment Setup", fontsize=8);
legend(loc=1);
xlabel("Lateral Position [mm]", fontsize=8);ylabel("Horizontal Position [mm]", fontsize=8);
xlim((0, 120)); ylim((120,0));

tight_layout()
savefig("blood_breast_setup.png", dpi=300, format="png")


######################################### Save results   ########################################
@save "breastexperiment.jld" grad_extend q_inv


