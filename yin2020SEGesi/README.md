# Extended source imaging -- a unifying framework for seismic & medical imaging

This repository reproduces the results presented in [*Extended source imaging -- a unifying framework for seismic & medical imaging*](https://doi.org/10.1190/segam2020-3426999.1), which is published in SEG Technical Program Expanded Abstract 2020.

## Dependencies

The minimum requirements for theis software, and tested version, are `Python 3.x` and `Julia 1.3.1`.
This software requires the following dependencies to be installed:

- [Devito](https://www.devitoproject.org), a python software as the finite-difference DSL and just-in-time compiler used for the wave equation operators. This can be installed with the command `pip install git+https://github.com/devitocodes/devito` or following the installation instructions for other platform or other modes of installation at [installation](http://devitocodes.github.io/devito/download.html)
- [DrWatson](https://juliadynamics.github.io/DrWatson.jl/dev/): a julia software in order to reproduce scientific project easily.

then configure python with the system one and rebuild the python calling poackage:

```
export PYTHON=$(which python)
julia -e 'using Pkg; Pkg.build("PyCall")'
```

## Software

This software is divided as follows:

*data/*:

 This directory contains the models used for both seismic imaging and medical imaging exmaples.
 
 
*script/*:

 This directory contains codes to run the corresponding experiments. You can run the followings to reproduce the experiments, also you can modify the file `gen_geometry.jl` to change the source/receiver setting and design your own experiment.
 
For example, you could use the commands below to replicate our result in the abstract 

```bash
$ cd script
$ julia Run_Seismic_Medical.jl
$ julia Run_Seismic_Medical_brain.jl
$ julia Run_Seismic_Medical_Breast.jl
```

## Citation

If you find this software useful in your research, we would appreciate it if you cite us as:

```bibtex
@CONFERENCE{yin2020SEGesi,
  author = {Ziyi Yin and Rafael Orozco and Philipp A. Witte and Mathias Louboutin and Gabrio
Rizzuti and Felix J. Herrmann},
  title = {Extended source imaging –- a unifying framework for seismic &
medical imaging},
  booktitle = {SEG Technical Program Expanded Abstracts},
  year = {2020},
  month = {09},
  pages = {3502-3506},
  abstract = {We present three imaging modalities that live on the crossroads of
seismic and medical imaging. Through the lens of extended source imaging, we
can draw deep connections among the fields of wave-equation based seismic and
medical imaging, despite first appearances. From the seismic perspective, we
underline the importance to work with the correct physics and spatially
varying velocity fields. Medical imaging, on the other hand, opens the
possibility for new imaging modalities where outside stimuli, such as laser
or radar pulses, can not only be used to identify endogenous optical or
thermal contrasts but that these sources can also be used to insonify the
medium so that images of the whole specimen can in principle be created.},
  keywords = {seismic imaging, medical imaging, variable projection, SEG},
  note = {(SEG, virtual)},
  doi = {10.1190/segam2020-3426999.1},
  url = {https://slim.gatech.edu/Publications/Public/Conferences/SEG/2020/yin2020SEGesi/yin2020SEGesi.html},
  presentation = {https://slim.gatech.edu/Publications/Public/Conferences/SEG/2020/yin2020SEGesi/yin2020SEGesi_pres.pdf},
  url2 = {https://slim.gatech.edu/Publications/Public/Conferences/SEG/2020/yin2020SEGesi/yin2020SEGesi_pres.mp4},
  software = {https://github.com/slimgroup/Software.SEG2020}
}
```

## Contact

For questions or issue, please contact Ziyi Yin: ziyi.yin@gatech.edu and Rafael Orozco: rorozco@gatech.edu
