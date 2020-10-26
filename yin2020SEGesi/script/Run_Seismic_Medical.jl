# Example for Variable Projection:
# Author: Ziyi Yin, Rafael Orozco, Philipp Witte, Mathias Louboutin, Gabrio Rizzuti, Felix J. Herrmann
# Date: April, 2020
#

using DrWatson
@quickactivate "ExtSrcImg"

include("gen_geometry.jl")

using PyPlot, FFTW, Images
using JUDI.TimeModeling, JOLI
using LinearAlgebra, Random, Statistics
using IterativeSolvers
using JLD

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
		W .= sqrt.((x.-zs_ave).^2f0.+(z.-zs_ave).^2f0)
		
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
			#random weights
			weights[xsrc_index[j], zsrc_index[j], 1, 1] = rand()
        	#weights[xsrc_index[j], zsrc_index[j], 1, 1] = 1f0
    	end
	else
		for j=1:nsrc
        	weights[xsrc_index[j], zsrc_index[j], 1, j] = 1f0
    	end
	end
    
    return weights
end

function GenLocRec(xrec, zrec, nrec, n)
	weights = zeros(Float32, n[1], n[2], 1, 1)

	#for Simultaneous we need to put all the weights in the same slice. 
	for j=1:nrec
        weights[Int(round(xrec[j]/d[1])), Int(round(zrec[j]/d[2])), 1, 1] = 1f0
    end
    
    return weights
end

function GenSimData(q₀; model=model)   # pass q₀ directly instead of z
    # Simultaneous observed data
    d_obs = F*vec(q₀)
    d_obs = reshape(d_obs, recGeometry.nt[1], nrec, 1, nsrc)  # reshape into tensor
    return d_obs
end


############################################ Models ###############################################

M = load("../data/sigsbee2A_model.jld")

m = M["m"][700:1900,100:800]                     # True model

N = load("../data/background_model.jld")	
m0 = (N["m0"])[700:1900,100:800]   	# background model

dm = m - m0
n = size(m)
d = M["d"]
o = M["o"]


model0 = Model(n, d, o, m0)
model = Model(n, d, o, m)

m0 = reshape(model0.m, n[1], n[2], 1, 1)
m = reshape(model.m, n[1], n[2], 1, 1)

extentx = n[1]*d[1] / 1000 #in km
extentz = n[2]*d[2] / 1000 #in km

# receiver sampling and recording time
timeR = 5500f0   # receiver recording time [ms]
dtR = 4f0    # receiver sampling interval
# Source sampling and number of time steps
timeS = 5500f0
dtS = 4f0

mode = "seismic"
#mode = "medical"

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

	# Set up receiver structure
	xrec, yrec, zrec, nrec, nsrc = GenRecGeometryMedical();
	recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

	# Set up source structure
	xsrc_index, zsrc_index, nsrc_simult = GenSrcGeometryMedical();

	#generate tensor with sources
	q           = GenSimSourceMulti(xsrc_index, zsrc_index, nsrc, model.n);

	#get muting region
	mute_mask = GenMuteMedical()

else
	error("Please choose a correct mode.")
end

figure(6); imshow(reshape(m, n[1], n[2])',extent=(0,extentx,extentz,0));
cb = colorbar(fraction=0.038, pad=0.04);
cb[:set_label]("Squared slowness [km^2/s^2]")
#PyPlot.scatter(xrec*d[1]/1000,zrec*d[1]/1000,marker=".",color="black",label="receivers");
PyPlot.scatter(xrec/1000,zrec/1000,marker=".",color="red",label="receivers");
#PyPlot.scatter(xsrc_index*d[1]/1000,zsrc_index*d[2]/1000,label="sources",marker=".",color="white");
legend(loc=1);
xlabel("Lateral position [km]");ylabel("Depth [km]"); 
fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)

PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
figure(); imshow(reshape(sqrt.(1f0./m), n[1], n[2])',extent=(0,extentx,extentz,0));
#cb = colorbar(fraction=0.027, pad=0.04,format="%.4f");
cb = colorbar(fraction=0.027, pad=0.04);
#cb[:set_label]("Velocity [km/s]", labelsize=20)
#cb[:set_label]("Squared slowness [km^2/s^2]")
PyPlot.scatter(xrec/1000,zrec/1000,marker=".",color="red",label="receivers",s=1);
PyPlot.scatter(xsrc_index*d[1]/1000,zsrc_index*d[2]/1000,label="sources",marker="x",color="white",s=1);
legend(loc=3,fontsize=8);
xlabel("Lateral position [km]", fontsize=8);ylabel("Horizontal position [km]", fontsize=8);
fig = PyPlot.gcf(); fig.set_size_inches(7.2, 4)
#savefig("seismic_setup.png", bbox_inches="tight")
savefig("seismic_setup.png")

# Setup wavelet
f0 = 0.015  # dominant frequency in [kHz]
wavelet = ricker_wavelet(timeS,dtS,f0)

######################################### Set up JUDI operators ###################################

# Set up info structure for linear operators
ntComp = get_computational_nt(recGeometry, model)
info   = Info(prod(model0.n), nsrc, ntComp)

# Write shots as array
opt = Options(return_array = true, subsampling_factor=20, isic=true)

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

#### 1.) RTM
J1=judiJacobian(F0,q)
res_rtm = d_obs-𝒢(q,m0);
grad_rtm=reshape(adjoint(J1)* vec(res_rtm),n[1],n[2])

#### 2.) Extended source gradient - inversion scheme
#Generate weight matrix W. Used in CG for source estimation
if mode == "medical"
	xs = reshape(xsrc_index*d[1],:,1,nsrc_simult);
	zs = reshape(zsrc_index*d[2],:,1,nsrc_simult); #z coords of sources
elseif mode == "seismic"
	xs = reshape(xsrc_index*d[1],:,1,nsrc);
	zs = reshape(zsrc_index*d[2],:,1,nsrc); #z coords of sources
end

W =  weight(n, d, xs, zs, nsrc, mode);

#run cg to invert for estimated source

q_iter = 5
λ      = 100
q_inv = zeros(Float32, n[1], n[2], 1, nsrc);
q_inv, r_end, rvec = CG(d_obs, q_inv, m0, W; maxiter=q_iter,λ=λ)


#get jacobian using estimated source and use it to get gradient
J = judiJacobian(F0, q_inv)
#grad_extend=reshape(adjoint(J)* vec(mask.*r_end),n[1],n[2])
grad_extend=reshape(adjoint(J)* vec(r_end),n[1],n[2])
#### 3.) Extended source gradient - adjoint scheme
q_adj = 𝒢_T(d_obs,m0)
J2=judiJacobian(F0,q_adj) #check with francis on this 
res_adj = d_obs-𝒢(q_adj,m0) 
grad_adj =reshape(adjoint(J2)* vec(res_adj),n[1],n[2])

######################################### Visualize results   ###################################

#show gradient images

dm = reshape(m-m0,n[1],n[2])
mute_end = zeros(Integer,n[1])
for i = 1:n[1]
	for j = 1:n[2]
		if dm[i,j] >= 5f-3
			mute_end[i] = j
			break
		end
	end
end
D = judiDepthScaling(model)

new_inv=reshape(model_topmute(n,mute_end,5,grad_extend),n[1],n[2]);
final_inv = reshape(D*D*D*D*D*D*vec(deepcopy(new_inv)),n[1],n[2])
PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
figure();imshow(final_inv'/norm(final_inv',Inf),vmin=-0.1,vmax=0.1,cmap="gray",extent=(0,extentx,extentz,0))
xlabel("Lateral position [km]", fontsize=8);ylabel("Horizontal position [km]", fontsize=8);
fig = PyPlot.gcf(); fig.set_size_inches(7.2, 4)
savefig("seismic_inv_imaging.png")

new_rtm=reshape(model_topmute(n,mute_end,5,grad_rtm),n[1],n[2]);
final_rtm = reshape(D*D*D*D*D*D*vec(deepcopy(new_rtm)),n[1],n[2])
PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
figure();imshow(final_rtm'/norm(final_rtm',Inf),vmin=-0.1,vmax=0.1,cmap="gray",extent=(0,extentx,extentz,0))
xlabel("Lateral position [km]", fontsize=8);ylabel("Horizontal position [km]", fontsize=8);
fig = PyPlot.gcf(); fig.set_size_inches(7.2, 4)
savefig("seismic_rtm_imaging.png")

new_adj=reshape(model_topmute(n,mute_end,5,grad_adj),n[1],n[2]);
final_adj = reshape(D*D*D*D*D*D*vec(deepcopy(new_adj)),n[1],n[2])
PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
figure();imshow(final_adj'/norm(final_adj',Inf),vmin=-0.1,vmax=0.1,cmap="gray",extent=(0,extentx,extentz,0))
cb = colorbar(fraction=0.027, pad=0.04);
"Horizontal position [km]", fontsize=8);
fig = PyPlot.gcf(); fig.set_size_inches(5.6, 3.5)
savefig("seismic_adj_imaging.png", bbox_inches="tight")

if mode == "seismic"	# one single extended source in the middle

	q_inv_stack=sum(q_inv,dims=4)[:,:,1,1];
	figure();imshow(q_inv_stack',vmin=-2f-3,vmax=2f-3,extent=(0,extentx,extentz,0));title("Extended Source from Inversion");
	xlabel("Lateral position [samples]");ylabel("Depth [samples]");
	colorbar(fraction=0.028, pad=0.04);
	fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)

	q_adj_stack=sum(q_adj,dims=4)[:,:,1,1];
	figure();imshow(q_adj_stack',vmin=-40000,vmax=40000,extent=(0,extentx,extentz,0));title("Extended Source from Adjoint");
	xlabel("Lateral position [samples]");ylabel("Depth [samples]");
	colorbar(fraction=0.028, pad=0.04);
	fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)

	if nsrc != 1
		mid_loc = Int(round(nsrc/2)+1)
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
	#show estimated sources
	q_adj_stack=sum(q_adj,dims=4)[:,:,1,1];
	figure(70);imshow(q_adj_stack',extent=(0,extentx,extentz,0),vmin=-40000,vmax=40000);title("Extended Source from Adjoint");
	xlabel("Lateral position [km]");ylabel("Depth [km]");
	colorbar(fraction=0.028, pad=0.04);
	fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)

	q_inv_stack=sum(q_inv,dims=4)[:,:,1,1];
	figure(8);imshow(q_inv_stack',extent=(0,extentx,extentz,0),vmin=-2f-3,vmax=2f-3);title("Extended Source from Inversion");
	xlabel("Lateral position [km]");ylabel("Depth [km]");
	colorbar(fraction=0.028, pad=0.04);
	fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)

	#show estimated sources
	q_adj_stack=sum(q_adj,dims=4)[:,:,1,1];
	figure(70);imshow(q_adj_stack',extent=(0,extentx,extentz,0));title("Extended Source from Adjoint");
	xlabel("Lateral position [km]");ylabel("Depth [km]");
	colorbar(fraction=0.028, pad=0.04);
	fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)

	q_inv_stack=sum(q_inv,dims=4)[:,:,1,1];
	figure(8);imshow(q_inv_stack',extent=(0,extentx,extentz,0));title("Extended Source from Inversion");
	xlabel("Lateral position [km]");ylabel("Depth [km]");
	colorbar(fraction=0.028, pad=0.04);
	fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)

	q_stack=sum(q,dims=4)[:,:,1,1];
	figure(19);imshow(q_stack',extent=(0,extentx,extentz,0));title("Real Source");
	xlabel("Lateral position [km]");ylabel("Depth [km]");
	colorbar(fraction=0.028, pad=0.04);
	fig = PyPlot.gcf(); fig.set_size_inches(12.0, 7.0)
end


######################################### Save results   ########################################

#@save "1src_precondition.jld" grad_adj grad_rtm grad_extend q_inv q_adj m m0 mute_mask xrec yrec zrec xsrc_index zsrc_index rvec

