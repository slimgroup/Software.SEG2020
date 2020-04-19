using MAT
using PyPlot
using JOLI
using GenSPGL
using SeisJOLI
using Arpack
using LinearAlgebra

###load the weighted algorithm function
include("../script/Weighted_LR.jl") 

###load the true data
datafile = MAT.matopen("../data/Full.mat")
data_full = read(datafile,"Full_Data")
datafull = copy(data_full[:,1:355,1:355])
nff,ns,nr = size(datafull)

###loaded the jittered index for subsampling
sub_index = MAT.matopen("../data/ind.mat")
ind = read(sub_index,"ind")
close(sub_index)

###define operators subsampling and joMH(source-receiver => midpoint-offset)
RM = joKron(joRestriction(ns,dropdims(trunc.(Int,ind),dims=1);DDT = Complex{Float64}), joDirac(nr;DDT = Complex{Float64}))
MH = joSRtoCMO(nr,ns)

###define rank information if rank=r_sub(convertional weighted method)
rank = 85;
r_sub = 25;

###define the final result to save
Final_result = zeros(Complex{Float64},nr,ns,nff)
Final_SNR = zeros(Float64,nff,1)

###weighted LR to interpolate the data
for sliceNum = 31:304  #(round([7Hz,75Hz]/df+1=[30,304]))

    if sliceNum == 31 
	# load unweighted result as prior information
	pre_int = MAT.matopen(string("../data/Idx_", string(sliceNum-1),".mat"))
	pre_L = read(pre_int,"L1")
	pre_R = read(pre_int,"R1")
	close(pre_int)
	
	# using limited-subspace weighted method
    	SNR, Result, L1, R1 = Weighted_LR(pre_L, pre_R, datafull[sliceNum,:,:], RM, MH, rank, r_sub)
    	println("Num=",sliceNum)
    else 
	# load prior information(neighbor low frequency results)
	pre_int = MAT.matopen(string("../julia_result/Idx_", string(sliceNum-1),".mat"))
	pre_L = read(pre_int,"L1")
	pre_R = read(pre_int,"R1")
	close(pre_int)
	
	# using limited-subspace weighted method
	SNR, Result, L1, R1 = Weighted_LR(pre_L, pre_R, datafull[sliceNum,:,:], RM, MH, rank, r_sub)
    	println("Num=",sliceNum)
    end
    Final_SNR[sliceNum,1] = SNR;
    Final_result[:,:,sliceNum] = Result;
	
    # save each frequency result
    file = matopen(string("../julia_result/Idx_",string(sliceNum),".mat"), "w")
    write(file,string("Final_result",string(sliceNum)),Result)
    write(file,"SNR",SNR)
    write(file,"L1",L1)
    write(file,"R1",R1)
    close(file);
end

### save all limited-subspace weighted result
file = matopen("../julia_result/recursive_data.mat", "w")
write(file,"Final_result",Final_result)
write(file,"Final_SNR",Final_SNR)
close(file)
