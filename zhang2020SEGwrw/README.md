# Wavefield recovery with limited-subspace weighted matrix factorizations

This repository will contain codes for generating results in Zhang, Y., Sharan, S., Lopez, O, and Herrmann, F.J., Wavefield recovery with limited-subspace weighted matrix factorizations.

## Dependencies

The minimum requirements for theis software, and tested version, are `Python 3.x` and `Julia 1.2.0`.
This software requires the following dependencies to be installed:

- [MAT](https://github.com/JuliaIO/MAT.jl). This library can read MATLAB .mat files, both in the older v5/v6/v7 format, as well as the newer v7.3 format.
- [PyPlot](https://github.com/JuliaPy/PyPlot.jl), a plotting julia package based on `matplotlib`.
- [JOLI](https://github.com/slimgroup/JOLI.jl),Julia framework for constructing matrix-free linear operators with explicit domain/range type control and applying them in basic algebraic matrix-vector operations.
- [GenSPGL](https://github.com/slimgroup/GenSPGL.jl). A Julia solver for large scale minimization problems using any provided norm.
- [SeisJOLI](https://github.com/slimgroup/SeisJOLI.jl). Collection of SLIM in-house operators based on JOLI package.
- [Arpack](https://github.com/JuliaLinearAlgebra/Arpack.jl). Julia wrapper for the arpack library designed to solve large scale eigenvalue problems.
- [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/). Julia provides native implementations of many common and useful linear algebra operations which can be loaded with using LinearAlgebra. 

First, you need install the following packages from the stable master branch:
```
using Pkg; Pkg.add(PackageSpec(url="https://github.com/slimgroup/JOLI.jl"))
using Pkg; Pkg.add(PackageSpec(url="https://github.com/slimgroup/SeisJOLI.jl"))
using Pkg; Pkg.add(PackageSpec(url="https://github.com/slimgroup/GenSPGL.jl"))
```

then install dependencies:
```
Pkg.add("MAT")
Pkg.add("PyPlot")
Pkg.add("Arpack")
Pkg.add("LinearAlgebra")
```

## Software
This software is divided as follows:

 The ground truth data collected from the Gulf of Suez [Full.mat](https://slim.gatech.edu/PublicationsData/zhang2020SEGwrw/).

*data/*:
 
 This directory contains Jittered subsampling indexes[ind.mat] and one unweighted initial prior information[Idx_30.mat].
 
*script/*: 

 This directory contains codes to run the corresponding experiments.You can run the 'RecursiveLR.jl' to reproduce the experiments, also you can modify the files to change the settings and design your own experiment.
 
 ```
 RecursiveLR.jl #The main function of our experiments
 Weighted_LR.jl #The subfunction of limited-subspace weighted method for each frequency slice
 NLfunForward_test1.jl #The weighted forward function to implement the data misfit constraint
 ```
 
## Citation

If you find this software useful in your research, we would appreciate it if you cite:

```bibtex
@conference {zhang2020SEGwrw,
	title = {Wavefield recovery with limited-subspace weighted matrix factorizations},
	year = {2020},
	note = {Accepted in SEG},
	month = {4},
	abstract = {Modern-day seismic imaging and monitoring technology increasingly rely on dense full-azimuth sampling. Unfortunately, the costs of acquiring densely sampled data rapidly become prohibitive and we need to look for ways to sparsely collect data, e.g. from sparsely distributed ocean bottom nodes, from which we then derive densely sampled surveys through the method of wavefield reconstruction. Because of their relatively cheap and simple calculations, wavefield reconstruction via matrix factorizations has proven to be a viable and scalable alternative to the more generally used transform-based methods. While this method is capable of processing all full azimuth data frequency by frequency slice, its performance degrades at higher frequencies because monochromatic data at these frequencies is not as well approximated by low-rank factorizations. We address this problem by proposing a recursive recovery technique, which involves weighted matrix factorizations where recovered wavefields at the lower frequencies serve as prior information for the recovery of the higher frequencies. To limit the adverse effects of potential overfitting, we propose a limited-subspace recursively weighted matrix factorization approach where the size of the row and column subspaces to construct the weight matrices is constrained. We apply our method to data collected from the Gulf of Suez, and our results show that our limited-subspace weighted recovery method significantly improves the recovery quality.},
	keywords = {algorithm, data reconstruction, frequency-domain, interpolation, Processing, SEG},
	url = {https://slim.gatech.edu/content/wavefield-recovery-limited-subspace-weighted-matrix-factorizations},
	author = {Yijun Zhang, Shashin Sharan, Oscar Lopez, Felix J. Herrmann}
}
```

## Contact

For questions or issue, please contact yzhang3198@gatech.edu.


