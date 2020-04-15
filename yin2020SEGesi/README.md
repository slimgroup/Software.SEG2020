# Extended source imaging -- a unifying framework for seismic & medical imaging

This repository reproduces the results presented in the above named 2020 SEG abstract.


## Dependencies

The minimum requirements for theis software, and tested version, are `Python 3.x` and `Julia 1.3.1`.
This software requires the follwoing dependencies to be installed:

- [Devito](https://www.devitoproject.org), the finite-difference DSL and just-in-time compiler used for the TTI propagators. This can be installed with the command `pip install git+https://github.com/devitocodes/devio` or following the installation instructions for other platform or other modes of installation at [installation](http://devitocodes.github.io/devito/download.html)
- [JUDI](https://github.com/slimgroup/JUDI.jl), the julia linear algebra DSL built on top of Devito to generate the anisotropic data. Please follow the installation instruction at [JUDI](https://github.com/slimgroup/JUDI.jl) 
- [JOLI](https://github.com/slimgroup/JOLI.jl),Julia framework for constructing matrix-free linear operators with explicit domain/range type control and applying them in basic algebraic matrix-vector operations
- [SetIntersectionProjection.jl](https://github.com/slimgroup/SetIntersectionProjection.jl), the Julia software for computing projections onto intersections of convex and non-convex constraint sets, mainly used for tv-norm regularized inversion
- [JLD2](https://github.com/JuliaIO/JLD2.jl), a julia IO package used here for simpolicity
- [Images](https://github.com/JuliaImages/Images.jl), an image processing julia packsge used to add guassian kernel filters to velocity model
- [PyPlot](https://github.com/JuliaPy/PyPlot.jl), a plotting julia package based on `matplotlib`


The julia packages can be installed as follows once [Devito] is installed.
First, you need install JUDI with the TTI branch that is currently separated from the stable master branch:

```
julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/slimgroup/JOLI.jl"))'
julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/slimgroup/SegyIO.jl"))'
julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/slimgroup/JUDI.jl.git", rev="v1-tti"))'
julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/slimgroup/SetIntersectionProjection.jl.git"))'



```

then configure python with the system one and rebuild the python calling poackage:

```
export PYTHON=$(which python)
julia -e 'using Pkg; Pkg.build("PyCall")'
```

Finally install dependencies:

```
julia -e 'using Pkg; Pkg.add("JLD2")'
julia -e 'using Pkg; Pkg.add("IMAGES")'
julia -e 'using Pkg; Pkg.add("PyPlot")'
```

## Software

This software is divided as follows:

*data/*:

 This directory contains the models used for both seismic imaging and medical imaging exmaples.
 
 
*script/*:

 This directory contains codes to run the corresponding experiments. You can run the followings to reproduce the experiments, also you can modify the file `gen_geometry.jl` to change the source/receiver setting and design your own experiment.
 
```bash
$ cd script
$ julia Run_Seismic_Medical.jl
$ julia Run_Seismic_Medical_brain.jl
$ julia Run_Seismic_Medical_Breast.jl
```

## Contact

For questions or issue, please contact Ziyi Yin: ziyi.yin@gatech.edu and Rafael Orozco: rorozco@gatech.edu
