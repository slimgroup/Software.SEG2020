# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using SetIntersectionProjection
using JUDI.TimeModeling, SeisIO, JOLI, PyPlot, ProximalOperators, IterativeSolvers, JUDI.SLIM_optim, MAT, Dierckx, SpecialFunctions, LinearAlgebra, SparseArrays, FFTW
using Statistics, Random

## setup for bas
mutable struct compgrid
  d :: Tuple
  n :: Tuple
end

function spg_tv_norm(d_sim,objective_function,q_true, q_init, model; maxiter =5,tv_norm = 0.5)



  # with constraints:
  options=PARSDMM_options()
  options.FL=Float32
  options=default_PARSDMM_options(options,options.FL)
  options.adjust_gamma           = true
  options.adjust_rho             = true
  options.adjust_feasibility_rho = true
  options.Blas_active            = true
  options.maxit                  = 1000
  options.feas_tol= 0.001
  options.obj_tol=0.001
  options.evol_rel_tol = 0.00001

  options.rho_ini=[1.0f0]

  set_zero_subnormals(true)
  BLAS.set_num_threads(2)
  FFTW.set_num_threads(2)
  options.parallel=false
  options.feasibility_only = false
  options.zero_ini_guess=true

  #select working precision
  if options.FL==Float64
    TF = Float64
    TI = Int64
  elseif options.FL==Float32
    TF = Float32
    TI = Int32
  end

  #----# make projector
  #comp_grid=compgrid((44.75,44.75),(model.n[1],model.n[2]))
  comp_grid=compgrid((1,1),(model.n[1],model.n[2]))


  constraint = Vector{SetIntersectionProjection.set_definitions}()

  #TV
  (TV,dummy1,dummy2,dummy3)=get_TD_operator(comp_grid,"TV",options.FL)
  m_min     = 0.0
  m_max     = tv_norm*norm(TV*vec(q_true),1)
  set_type  = "l1"
  TD_OP     = "TV"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

  #print(0.5f0*norm(TV*vec(weights),1))


  #NEW
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)
  options.rho_ini        = ones(length(TD_OP))*10.0

  proj_intersection = x_new-> PARSDMM(x_new,AtA,TD_OP,set_Prop,P_sub,comp_grid,options)

  function prj!(input)
      (x, dummy1, dummy2, dymmy3) = proj_intersection(input)
      return x
  end

  #proj_q_adj = prj!(vec(q_norm))
  #imshow(reshape(proj_q_adj,n))

  # Source localization with SPG
  options_spg            = spg_options(verbose=3, maxIter=maxiter, memory=5)
  options_spg.progTol	   = 1e-60

  #options_spg = spg_options(verbose=3, maxIter=maxiter, memory=5, suffDec=1f-6)
  #options_spg.progTol=1e-60
  #options_spg.testOpt=false
  #options_spg.interp=2

  #ProjBound(x)       = x

  #print(length(w_init))
  q_spg, fsave, funEvals = minConf_SPG(objective_function, vec(q_init), prj!, options_spg)

  return q_spg
end
