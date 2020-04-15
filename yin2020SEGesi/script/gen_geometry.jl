### Example set up for seismic or medical experiments
### Author: Ziyi Yin* and Rafael Orozco*
### March, 2020


function GenMuteSeismic()
	##Muting to properly visualize
	mute_region = ones(Float32,n[1],n[2]);
	mute_region[:,1:60].=0;
	mute_mask = imfilter(mute_region, Kernel.gaussian(10));
	#q_mask_smooth = imfilter(reshape(q_stack,model.n), Kernel.gaussian(30))
	
    return mute_mask
end

function GenMuteMedical()
	##Muting to properly visualize

	#pading for receivers
	pad_rec = 10 + 10

	#muting due to recerivers
	mute_region = ones(Float32,n[1],n[2]);
	mute_region[:,1:100].=0;
	mute_region[:,600:end].=0;
	mute_region[1:100,:].=0;
	mute_region[1100:end,:].=0;
	mute_mask = imfilter(mute_region, Kernel.gaussian(10));
	

	q_stack=sum(q,dims=4)[:,:,1,1];
	#muting due to sources
	q_smooth = imfilter(q_stack[:,:,1,1], Kernel.gaussian(10))
	q_smooth[q_smooth .> 0.00001] .= 1

	q_smooth_2 = imfilter(q_smooth, Kernel.gaussian(10))
	q_smooth_2_flip = 1 .- q_smooth_2
	mute_mask_full = mute_mask .* q_smooth_2_flip


    return mute_mask_full
end

function GenMuteMedical_brain()
	##Muting to properly visualize

	#muting due to recerivers


	pad = 40
	mute_region = ones(Float32,n[1],n[2]);
	mute_region[:,1:pad].=0;
	mute_region[:,n[2]-pad:end].=0;
	mute_region[1:pad,:].=0;
	mute_region[n[1]-pad:end,:].=0;
	mute_mask = imfilter(mute_region, Kernel.gaussian(10));
	

	q_stack=sum(q,dims=4)[:,:,1,1];
	#muting due to sources
	q_smooth = imfilter(q_stack[:,:,1,1], Kernel.gaussian(20))
	q_smooth[q_smooth .> 0.00001] .= 1

	q_smooth_2 = imfilter(q_smooth, Kernel.gaussian(10))
	q_smooth_2_flip = 1 .- q_smooth_2
	mute_mask_full = mute_mask .* q_smooth_2_flip


    return mute_mask_full
end

function GenMuteMedical_breast()
	##Muting to properly visualize

	#muting due to recerivers
	rad     = 250
	centerx = 300
	centery = 300 
	mute_region = ones(Float32,n[1],n[2]);

	for i in 1:n[1]
		for j in 1:n[2]
			if (i-centerx)^2 + (j-centery)^2 > rad^2
				mute_region[i,j] = 0
			end
		end
	end
	
	mute_mask = imfilter(mute_region, Kernel.gaussian(10));

	q_stack=sum(q,dims=4)[:,:,1,1];
	#muting due to sources
	q_smooth = imfilter(q_stack[:,:,1,1], Kernel.gaussian(15))
	q_smooth[q_smooth .> 0.0001] .= 1

	q_smooth_2 = imfilter(q_smooth, Kernel.gaussian(5))
	q_smooth_2_flip = 1 .- q_smooth_2
	mute_mask_full = mute_mask .* q_smooth_2_flip


    return mute_mask_full
end

function GenRecGeometrySeismic()
	
	# Set up receiver geometry
	nrec = 500
	domain_x = (model.n[1] - 1)*model.d[1]
	xmin = 50f0*d[1]
	xmax = domain_x - xmin
	xrec = range(xmin, stop=xmax, length=nrec)
	yrec = 0f0
	zrec = range(30f0*d[2], 30f0*d[2], length=nrec)
	nsrc = 90
	# Set up receiver structure
	
    return xrec,yrec,zrec,nrec,nsrc
end

function GenRecGeometryMedical()

	nsrc = 1
	pad = 50f0

	domain_x = (model0.n[1] - 1)*model0.d[1]
	domain_z = (model0.n[2] - 1)*model0.d[2]

	xmin = pad*d[1]
	xmax = domain_x - xmin

	zmin = pad*d[2]
	zmax = domain_z - zmin

	#for sampling distance:
	#we put 500 receivers on top and bottom then calculate the sampling distance
	#for sides. 
	nrec_top_bottom = 500
	sampling_dist = (domain_x*model0.d[1]/(nrec_top_bottom-1))
	nrec_side = Int(round(domain_z*model0.d[1] / sampling_dist))
	

	#lateral x distance
	xrec_top    = range(xmin, stop=xmax, length=nrec_top_bottom)
	xrec_bottom = range(xmin, stop=xmax, length=nrec_top_bottom)
	xrec_left   = range(xmin, stop=xmin, length=nrec_side)
	xrec_right  = range(xmax, stop=xmax, length=nrec_side)
	
	#depth z distance
	zrec_top    = range(zmin,   zmin, length=nrec_top_bottom)
	zrec_bottom = range(zmax,   zmax, length=nrec_top_bottom)
	zrec_left   = range(zmin,   zmax, length=nrec_side)
	zrec_right  = range(zmin,   zmax, length=nrec_side)

	yrec = 0f0 #2d so always 0

	xrec = vcat(xrec_top,xrec_bottom,xrec_left,xrec_right)
	zrec = vcat(zrec_top,zrec_bottom,zrec_left,zrec_right)

	#xrec = vcat(xrec_top)
	#zrec = vcat(zrec_top)

	nrec = size(xrec)[1]    # total no. of receivers

	
    return xrec, yrec, zrec, nrec, nsrc
end

function GenSrcGeometrySeismic(nsrc)
	
	# Multiple source locations (locations as indices)
	if nsrc == 1
		xsrc_index = Int.(round.((range(800,stop=800,length=nsrc))))   # indices where sim source weights are 1
		zsrc_index = Int.(round.((range(30,stop=30,length=nsrc))))
		return xsrc_index,zsrc_index
	end
	xsrc_index = Int.(round.((range(50,stop=1150,length=nsrc))))   # indices where sim source weights are 1
	zsrc_index = Int.(round.((range(30,stop=30,length=nsrc))))

    return xsrc_index,zsrc_index
end

function GenSrcGeometryMedical()
	nsrc_simult = 1

	# Multiple source locations (locations as indices)
	#xsrc_index = Int.(round.((range(50,stop=1150,length=nsrc_simult))))   # indices where sim source weights are 1
	#xsrc_index = Int.(round.((range(300,stop=1150,length=nsrc_simult))))   # indices where sim source weights are 1
	
	xsrc_index = Int.(round.((range(800,stop=800,length=nsrc_simult))))
	zsrc_index = Int.(round.((range(550,stop=550,length=nsrc_simult))))
	# load data and associated grid Array


	# file         = matopen(input_path*"NIC_400_full_Phantom_360.mat");
	# p0           = read(file, "p0");

	# size_rows = 501
	# size_cols = 501
	# p0_resize = imresize(p0, (size_rows, size_cols))
	# p0_resize[p0_resize .> 0] .= 1.0

	# init_x = 100
	# init_z = 200
	# in_model = zeros(Float32,n[1],n[2]);
	# in_model[init_x:init_x+size_rows-1,init_z:init_z+size_cols-1] .= p0_resize
	# #imshow(in_model')

	# # load source locations
	# inds         = findall(x->x>0.0,in_model);


	# xsrc_index = convert(Array{Int64,1},zeros(size(inds)));
	# zsrc_index = convert(Array{Int64,1},zeros(size(inds)));

	# for i=1:length(inds)
	#        xsrc_index[i] = inds[i][1];
	#        zsrc_index[i] = inds[i][2];
	# end

	# nsrc_simult = size(xsrc_index)[1]

    return xsrc_index, zsrc_index, nsrc_simult
end

function Gen_Exp_Medical_Skull_Phantom_Source()
	########----Setup receivers
	nsrc = 1

	domain_x = (model0.n[1] - 1)*model0.d[1]
	domain_z = (model0.n[2] - 1)*model0.d[2]

	
	file         = matopen("NIC_441_full_Phantom_60.mat");
	p0           = read(file, "p0");

	size_rows = 100
	size_cols = 100
	p0_resize = imresize(p0, (size_rows, size_cols))
	p0_resize[p0_resize .> 0] .= 1.0

	init_x = 80
	init_z = 100
	in_model = zeros(Float32,n[1],n[2]);
	in_model[init_x:init_x+size_rows-1,init_z:init_z+size_cols-1] .= p0_resize
	#imshow(in_model')

	# load source locations
	inds         = findall(x->x>0.0,in_model);


	xsrc_index = convert(Array{Int64,1},zeros(size(inds)));
	zsrc_index = convert(Array{Int64,1},zeros(size(inds)));

	for i=1:length(inds)
	       xsrc_index[i] = inds[i][1];
	       zsrc_index[i] = inds[i][2];
	end

	nsrc_simult = size(xsrc_index)[1]

	# load data and associated grid Array

	# file         = matopen(input_path*"NIC_400_full_Phantom_360.mat");
	# p0           = read(file, "p0");

	# size_rows = 501
	# size_cols = 501
	# p0_resize = imresize(p0, (size_rows, size_cols))
	# p0_resize[p0_resize .> 0] .= 1.0

	# init_x = 100
	# init_z = 200
	# in_model = zeros(Float32,n[1],n[2]);
	# in_model[init_x:init_x+size_rows-1,init_z:init_z+size_cols-1] .= p0_resize
	# #imshow(in_model')

	# # load source locations
	# inds         = findall(x->x>0.0,in_model);

	# xsrc_index = convert(Array{Int64,1},zeros(size(inds)));
	# zsrc_index = convert(Array{Int64,1},zeros(size(inds)));

	# for i=1:length(inds)
	#        xsrc_index[i] = inds[i][1];
	#        zsrc_index[i] = inds[i][2];
	# end

	# nsrc_simult = size(xsrc_index)[1]

	nrec = 512

	function circleShape(h, k ,r)
		theta = LinRange(0,2*pi, nrec)
		h .+ r*sin.(theta), k .+ r*cos.(theta)
	end

	xrec, zrec = circleShape(domain_x / 2,domain_z/2, domain_z/2-5*d[2])
	yrec = 0f0 #2d so always 0
	
	s=rand(1:512,100);
	xrec_rand = xrec[s];
	zrec_rand = zrec[s];
	nrec = 100


    return xrec_rand, yrec, zrec_rand, nrec, nsrc, xsrc_index, zsrc_index, nsrc_simult
#	return xrec, yrec, zrec, nrec, nsrc, xsrc_index, zsrc_index, nsrc_simult
end

function Gen_Exp_Medical_Skull_Single_Source()
	########----Setup receivers
	nsrc = 1
	nsrc_simult=1

	domain_x = (model0.n[1] - 1)*model0.d[1]
	domain_z = (model0.n[2] - 1)*model0.d[2]

	
	xsrc_index = Int.(round.((range(128,stop=128,length=nsrc_simult))))
	zsrc_index = Int.(round.((range(128,stop=128,length=nsrc_simult))))
	# load data and associated grid Array

	# file         = matopen(input_path*"NIC_400_full_Phantom_360.mat");
	# p0           = read(file, "p0");

	# size_rows = 501
	# size_cols = 501
	# p0_resize = imresize(p0, (size_rows, size_cols))
	# p0_resize[p0_resize .> 0] .= 1.0

	# init_x = 100
	# init_z = 200
	# in_model = zeros(Float32,n[1],n[2]);
	# in_model[init_x:init_x+size_rows-1,init_z:init_z+size_cols-1] .= p0_resize
	# #imshow(in_model')

	# # load source locations
	# inds         = findall(x->x>0.0,in_model);

	# xsrc_index = convert(Array{Int64,1},zeros(size(inds)));
	# zsrc_index = convert(Array{Int64,1},zeros(size(inds)));

	# for i=1:length(inds)
	#        xsrc_index[i] = inds[i][1];
	#        zsrc_index[i] = inds[i][2];
	# end

	# nsrc_simult = size(xsrc_index)[1]

	nrec = 512

	function circleShape(h, k ,r)
		theta = LinRange(0,2*pi, nrec)
		h .+ r*sin.(theta), k .+ r*cos.(theta)
	end

	xrec, zrec = circleShape(domain_x / 2,domain_z/2, domain_z/2-5*d[2])
	yrec = 0f0 #2d so always 0


    return xrec, yrec, zrec, nrec, nsrc, xsrc_index, zsrc_index, nsrc_simult
end


function Gen_Exp_Medical_Calc_Single_Source()


	########----Setup receivers
	nsrc = 1
	pad = 10f0

	domain_x = (model0.n[1] - 1)*model0.d[1]
	domain_z = (model0.n[2] - 1)*model0.d[2]

	xmin = pad*d[1]
	xmax = domain_x - xmin

	zmin = pad*d[2]
	zmax = domain_z - zmin

	#for sampling distance:
	#we put 500 receivers on top and bottom then calculate the sampling distance
	#for sides. 
	nrec_top_bottom = 50
	sampling_dist = (domain_x*model0.d[1]/(nrec_top_bottom-1))
	nrec_side = Int(round(domain_z*model0.d[1] / sampling_dist))
	

	#lateral x distance
	xrec_top    = range(xmin, stop=xmax, length=nrec_top_bottom)
	xrec_bottom = range(xmin, stop=xmax, length=nrec_top_bottom)
	xrec_left   = range(xmin, stop=xmin, length=nrec_side)
	xrec_right  = range(xmax, stop=xmax, length=nrec_side)
	
	#depth z distance
	zrec_top    = range(zmin,   zmin, length=nrec_top_bottom)
	zrec_bottom = range(zmax,   zmax, length=nrec_top_bottom)
	zrec_left   = range(zmin,   zmax, length=nrec_side)
	zrec_right  = range(zmin,   zmax, length=nrec_side)

	yrec = 0f0 #2d so always 0

	xrec = vcat(xrec_top,xrec_bottom,xrec_left,xrec_right)
	zrec = vcat(zrec_top,zrec_bottom,zrec_left,zrec_right)

	#xrec = vcat(xrec_top)
	#zrec = vcat(zrec_top)

	nrec = size(xrec)[1]    # total no. of receivers


	########----Setup source

	#this experiment puts a single under a large wall of high velocity (bone)

	nsrc_simult = 1

	# Multiple source locations (locations as indices)
	#xsrc_index = Int.(round.((range(50,stop=1150,length=nsrc_simult))))   # indices where sim source weights are 1
	#xsrc_index = Int.(round.((range(300,stop=1150,length=nsrc_simult))))   # indices where sim source weights are 1
	
	xsrc_index = Int.(round.((range(300,stop=300,length=nsrc_simult))))
	zsrc_index = Int.(round.((range(300,stop=300,length=nsrc_simult))))
	
	# load data and associated grid Array


	# file         = matopen(input_path*"NIC_400_full_Phantom_360.mat");
	# p0           = read(file, "p0");

	# size_rows = 501
	# size_cols = 501
	# p0_resize = imresize(p0, (size_rows, size_cols))
	# p0_resize[p0_resize .> 0] .= 1.0

	# init_x = 100
	# init_z = 200
	# in_model = zeros(Float32,n[1],n[2]);
	# in_model[init_x:init_x+size_rows-1,init_z:init_z+size_cols-1] .= p0_resize
	# #imshow(in_model')

	# # load source locations
	# inds         = findall(x->x>0.0,in_model);


	# xsrc_index = convert(Array{Int64,1},zeros(size(inds)));
	# zsrc_index = convert(Array{Int64,1},zeros(size(inds)));

	# for i=1:length(inds)
	#        xsrc_index[i] = inds[i][1];
	#        zsrc_index[i] = inds[i][2];
	# end

	# nsrc_simult = size(xsrc_index)[1]

    return xrec, yrec, zrec, nrec, nsrc, xsrc_index, zsrc_index, nsrc_simult
end



function Gen_Exp_Medical_Calc_Breast_Single()
	########----Setup source
	nsrc = 1
	nsrc_simult = 1

	# Artificial source location (locations as indices)
	xsrc_index = Int.(round.((range(300,stop=300,length=nsrc_simult))))
	zsrc_index = Int.(round.((range(300,stop=300,length=nsrc_simult))))


	######----Artificial source location (locations as indices)
	#blood_inds =findall(v[x->x == 1584.0])
	#xsrc_index = blood_inds[1,:]
	#zsrc_index = blood_inds[2,:]

	
	######------SETUP RING TRANSDUCER ARRAY
	domain_x = (model0.n[1] - 1)*model0.d[1]
	domain_z = (model0.n[2] - 1)*model0.d[2]

	nrec = 512
	function circleShape(h, k ,r)
		theta = LinRange(0,2*pi, nrec)
		h .+ r*sin.(theta), k .+ r*cos.(theta)
	end

	xrec, zrec = circleShape(domain_x / 2,domain_z/2, domain_z/2-1600)
	yrec = 0f0 #2d so always 0


    return xrec, yrec, zrec, nrec, nsrc, xsrc_index, zsrc_index, nsrc_simult
end


function Gen_Exp_Medical_Calc_Breast_Multiple()
	########----Setup source
	nsrc = 1
	nsrc_simult = 2

	# Artificial source location (locations as indices)
	xsrc_index = Int.(round.((range(200,stop=400,length=nsrc_simult))))
	zsrc_index = Int.(round.((range(300,stop=300,length=nsrc_simult))))

	#xsrc_index = vcat(xsrc_index, Int.(round.((range(300,stop=300,length=nsrc_simult)))))
	#zsrc_index = vcat(zsrc_index, Int.(round.((range(200,stop=400,length=nsrc_simult)))))

	######----Artificial source location (locations as indices)
	#blood_inds =findall(v[x->x == 1584.0])
	#xsrc_index = blood_inds[1,:]
	#zsrc_index = blood_inds[2,:]

	
	######------SETUP RING TRANSDUCER ARRAY
	domain_x = (model0.n[1] - 1)*model0.d[1]
	domain_z = (model0.n[2] - 1)*model0.d[2]

	nrec = 512
	function circleShape(h, k ,r)
		theta = LinRange(0,2*pi, nrec)
		h .+ r*sin.(theta), k .+ r*cos.(theta)
	end

	xrec, zrec = circleShape(domain_x / 2,domain_z/2, domain_z/2-1600)
	yrec = 0f0 #2d so always 0

	
	


    return xrec, yrec, zrec, nrec, nsrc, xsrc_index, zsrc_index, nsrc_simult
end

function Gen_Exp_Seismic_Calc_Breast_Multiple()
	########----Setup source
	nsrc = 4
	nsrc_simult = nsrc

	# Artificial source location (locations as indices)
	#xsrc_index = []
	#zsrc_index = []

	for i in 1:nsrc
		xsrc_index = vcat(xsrc_index, rand(200:400))
		zsrc_index = vcat(zsrc_index, rand(200:400))
	end
	#xsrc_index = Int.(round.((range(200,stop=400,length=nsrc_simult))))
	#zsrc_index = Int.(round.((range(300,stop=300,length=nsrc_simult))))

	#xsrc_index = vcat(xsrc_index, Int.(round.((range(300,stop=300,length=nsrc_simult)))))
	#zsrc_index = vcat(zsrc_index, Int.(round.((range(200,stop=400,length=nsrc_simult)))))

	######----Artificial source location (locations as indices)
	#blood_inds =findall(v[x->x == 1584.0])
	#xsrc_index = blood_inds[1,:]
	#zsrc_index = blood_inds[2,:]

	
	######------SETUP RING TRANSDUCER ARRAY
	domain_x = (model0.n[1] - 1)*model0.d[1]
	domain_z = (model0.n[2] - 1)*model0.d[2]

	nrec = 512
	function circleShape(h, k ,r)
		theta = LinRange(0,2*pi, nrec)
		h .+ r*sin.(theta), k .+ r*cos.(theta)
	end

	xrec, zrec = circleShape(domain_x / 2,domain_z/2, domain_z/2-1600)
	yrec = 0f0 #2d so always 0

    return xrec, yrec, zrec, nrec, nsrc, xsrc_index, zsrc_index, nsrc_simult
end

function Gen_Exp_Medical_Calc_Breast_Blood()
	########----Setup source
	nsrc = 1
	nsrc_simult = 1

	# Artificial source location (locations as indices)
	#xsrc_index = Int.(round.((range(300,stop=300,length=nsrc_simult))))
	#zsrc_index = Int.(round.((range(300,stop=300,length=nsrc_simult))))


	######----Artificial source location (locations as indices)
	#blood_inds =findall(x->x == 1584.0,v)

	#xsrc_index = getindex.(blood_inds, 1)
	#zsrc_index = getindex.(blood_inds, 2)

	####MANUALLY ADD SINGLE SOURCES
	zsrc_index = [428,495,539, 309, 342, 300, 108]
	xsrc_index = [392,380,213, 403, 445, 494, 413]
	nsrc_simult = 7
	
	######------SETUP RING TRANSDUCER ARRAY
	domain_x = (model0.n[1] - 1)*model0.d[1]
	domain_z = (model0.n[2] - 1)*model0.d[2]

	nrec = 512
	function circleShape(h, k ,r)
		theta = LinRange(0,2*pi, nrec)
		h .+ r*sin.(theta), k .+ r*cos.(theta)
	end

	xrec, zrec = circleShape(domain_x / 2,domain_z/2, domain_z/2-1600)

	yrec = 0f0 #2d so always 0


    return xrec, yrec, zrec, nrec, nsrc, xsrc_index, zsrc_index, nsrc_simult
end

function Gen_Exp_Seismic_Calc_Breast_Blood()
	########----Setup source
	nsrc = 7
	nsrc_simult = 7

	# Artificial source location (locations as indices)
	#xsrc_index = Int.(round.((range(300,stop=300,length=nsrc_simult))))
	#zsrc_index = Int.(round.((range(300,stop=300,length=nsrc_simult))))


	######----Artificial source location (locations as indices)
	# blood_inds =findall(x->x == 1584.0,v)

	# xsrc_index = getindex.(blood_inds, 1)
	# zsrc_index = getindex.(blood_inds, 2)

	####MANUALLY ADD SINGLE SOURCES
	#zsrc_index = [428]
	#xsrc_index = [392]
	zsrc_index = [428,495,539, 309, 342, 300, 108]
	xsrc_index = [392,380,213, 403, 445, 494, 413]
	nsrc_simult = 7
	
	######------SETUP RING TRANSDUCER ARRAY
	domain_x = (model0.n[1] - 1)*model.d[1]
	domain_z = (model0.n[2] - 1)*model.d[2]

	nrec = 512
	function circleShape(h, k ,r)
		theta = LinRange(0,2*pi, nrec)
		h .+ r*sin.(theta), k .+ r*cos.(theta)
	end

	xrec, zrec = circleShape(domain_x / 2,domain_z/2, domain_z/2)
	
	yrec = 0f0 #2d so always 0


    return xrec, yrec, zrec, nrec, nsrc, xsrc_index, zsrc_index, nsrc_simult
end


