# Time-domain wavefield reconstruction inversion in a TTI medium

This repository reproduces the results presented in the above named 2020 SEG abstract.


## Dependencies

The minimum requirements for theis software, and tested version, are `Python 3.x` and Julia 1.3.1`.
This software requires the follwoing dependencies to be installed:

- [Devito], the finite-difference DSL and just-in-time compiler used for the TTI propagators. This can be installed with the command `pip install git+https://github.com/devitocodes/devio` or following the installation instructions for other platform or other modes of installation at [installation](http://devitocodes.github.io/devito/download.html).
- [JUDI], the julia linear algebra DSL built on top of Devito to generate the anisotropic data. Please follow the installation instruction at [JUDI]
- [JLD2], a julia IO package used here for simpolicity
- [Images], an image processing julia packsge used to smooth the anisotropy parameter and velocity model
- [PyPlot], a plotting julia package based on `matplotlib`


The julia packages can be installed as follows once [Devito] is installed. First install JUDI:

```
julia -e 'using Pkg; Pkg.add("https://github.com/slimgroup/JUDI.jl.git")'
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

[Devito]:htttp;//github.com/devitocodes/devito
[JUDI]:http://github.com/slimgroup/JUDI.jl
[JLD2]:https://github.com/JuliaIO/JLD2.jl
[Images]:https://github.com/JuliaImages/Images.jl
[PyPlot]:https://github.com/JuliaPy/PyPlot.jl

## Software

This software is divided as follows:

*data/*:

 This directory contains  the velocity model and Thomsen parameters for the BG compass model in `bg_tti.jld` in the julia `JLD2` format. This directory then contains two script to generate the data required for the gradient computations. Please run bothh of these script from this directory:
```bash
$ julia data/gen_data_bg.jl
$ julia data/gen_data_GaussiaLens.jl
```

Onc e these scripts have run, you should see two data files called `GaussLens_data_tti.jld` and `BGCompass_data_tti.jld` in that data folder.

*src/*:

THis directory contains the source software for time-domain WRI, for both the acoustic case and TTI case. This software can also be used outside the scope of the examples made available here following thee calling convention used in our examples.

*scripts/*:

This folder contains four script, two to generate the results and two to plot these results.
- `gaussian_lens.jl` computes the gradient for the three different test case for the Gaussian lens model (true tti, error in tti and acoustic) and `plot_gaussian_lens.jl` generates the figures of the paper. These results are saved in the data folder in the `JLD2` format.

- `bg_compass.jl` generates the six different gradient for both FWI and WRI (twelve total) for the three different test case and two different water layer configurations and `plot_bg_compass.jl` generates the figures of the paper. These results are saved in the data folder in the `JLD2` format.


# Running the examples

To run the examples there is a couple options. Each julia script in *scritps* and *data* are standalone and can be run by themselves with `julia scripts/gaussian_lens.jl` for example. The ones in *scripts* will needne the data to have been already generated. For simplicity a script is provided to run all the examples at once with:

```
./scripts/run_all.sh

```

This script also accept input if you wish to only run one of the two models:

```
./scripts/run_all.sh BG
./scripts/run_all.sh GL
```


## Contact

For questions or issue, please contact Mathias Loubiutin: mlouboutin3@gatech.edu