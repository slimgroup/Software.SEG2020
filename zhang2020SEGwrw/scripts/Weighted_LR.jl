export Weighted_LR

using DrWatson
@quickactivate "LimitedWeightedLR"
using JOLI
using GenSPGL

using SeisJOLI
using Arpack
using LinearAlgebra

include(projectdir()*"/scripts/NLfunForward_test1.jl") ##include the weighted formward function

function Weighted_LR(Pre_L::Tx, Pre_R::Tx, Df_test::Tx, RM::joKron, MH::joLinearFunction, rank::Int, r_sub::Int)where {ETx<:Number, Tx<:AbstractMatrix{ETx}}
#Pre_L: prior information of L
#Pre_R: prior information of R
#Df_test: ground truth data which is used to subsampling and calculate the signal noise ratio 
#RM: subsampling operator
#MH: source-receiver <=> midpoint - offset
#rank: rank information
#r_sub: rank for limited subspaces 

nr,ns = size(Df_test)

### created subsampling data
subD = reshape(RM'*(RM*vec(Df_test)),nr,ns)
    
##transfer subsampling data to midoff domain
midD = reshape(MH*subD[:], nr, 2*ns-1)
b = copy(midD)
  	
###prior information
Pre_data = Pre_L*Pre_R';

U_pre = svd(Pre_L).U[:,1:r_sub];
V_pre = svd(Pre_R).U[:,1:r_sub];


LInit   = Pre_L;
RInit   = Pre_R;
initfact = 1
xinit   = initfact*[vec(copy(LInit));vec(copy(RInit))]

###set relative parameters
tau = norm(copy(xinit[:]),1)
sigmafact = 1e-2
sigma   = sigmafact*norm(copy(b[:]),2);

###define the parameters for params
afunT(x) = reshape(x[:],nr,2*ns-1)
params = Dict("nr"=>rank,  # define rank value here,
	      "nr1"=>r_sub,
              "Ind"=> vec(midD) .== 0,
              "numr"=> nr,
              "numc"=> 2*ns-1,
              "funForward"=> NLfunForward_test1,
              "afunT"=> afunT,
              "afun"=> afun,
              "mode"=> 1,
              "ls"=> 1,
              "logical"=> 0,
              "funPenalty"=> funLS,
              "w"=>0.65,  # define weighted scale if w = 1(unweighted method)
              "weight"=>[vec(copy(U_pre));vec(copy(V_pre))])
    
### Choose the parameters for GenSPGL
opts = spgOptions(optTol = 1e-5,
                  bpTol = 1e-5,
                  decTol = 1e-4,
                  project = TraceNorm_project,
                  primal_norm = TraceNorm_primal,
                  dual_norm = TraceNorm_dual,
                  proxy = true,
                  ignorePErr = true,
                  iterations = 150,
                  verbosity = 1,
		  funCompositeR = GenSPGL.funCompR1)

### using spgl1 to implenment the algorithm
xLS_jl, r, g, info = spgl1(NLfunForward_test1, b[:], x = xinit[:], tau = tau,
                                        sigma = sigma, options = opts, params = params)

### extracted the results
L_Num = params["numr"]*params["nr"]
L1 = xLS_jl[1:L_Num]
R1 = xLS_jl[L_Num+1:end]
L = reshape(L1,params["numr"],params["nr"]);
R = reshape(R1,params["numc"],params["nr"]);

### introduce barred quantities to recover the optimal solution
w = params["w"];
e1 = params["numr"]*params["nr1"];
W_data = params["weight"]
U_t = copy(W_data[1:e1,1])
V_t = copy(W_data[e1+1:end,1])
U = reshape(copy(U_t),params["numr"], params["nr1"])
V = reshape(copy(V_t),params["numc"], params["nr1"])

 PU = U*U'
 PV = V*V'
 m_U,n_U = size(PU)
 m_V,n_V = size(PV)

 PU_org = Matrix{Complex{Float64}}(I,m_U,n_U)-copy(PU)
 PV_org = Matrix{Complex{Float64}}(I,m_V,n_V)-copy(PV)
    
 QU = (1/w)*PU+PU_org
 QV = (1/w)*PV+PV_org

L = QU*L;
R = QV*R;

### obtain the final results
Result = reshape(MH'*vec(L*R'), nr,ns)

### calculate the SNR 
SNR = -20*log10(norm(vec(Df_test)-vec(Result))/norm(vec(Df_test)));

println("SNR = ",SNR)

return SNR,Result,L,R

end

