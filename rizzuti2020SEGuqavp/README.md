# Parameterizing uncertainties with deep invertible networks, an application to reservoir characterization

This repository reproduces the results presented in the corresponding SEG2020 abstract.


## Dependencies

This software has been tested with `Julia 1.5.1`, and requires the following dependencies to be installed:

- [DrWatson]: scientific project assistant for reproducibility, see `https://juliadynamics.github.io/DrWatson.jl/dev/`
- [Flux]: deep learning package
- [InvertibleNetworks]: SLIM package for invertible networks, see `https://github.com/slimgroup/InvertibleNetworks.jl`
- [Optim]: non-linear optimization tools, see `https://julianlsolvers.github.io/Optim.jl/stable/#`


# Running the examples

For the Gaussian mixture sampling estimation, run the command from the parent directory:
```
julia ./scripts/GaussMixture/VI_gaussmixture.jl
```

For running uncertainty quantification for full-waveform inversion on the Sleipner dataset, seismic data need be generated first by running:
```
julia ./scripts/Sleipner/gendata_Sleipner.jl
```
Deterministic FWI should then be run by:
```
julia ./scripts/Sleipner/FWI_Sleipner.jl
```
Finally, for training uncertainty quantification:
```
julia ./scripts/Sleipner/VI_Sleipner.jl
```

## Citation

For citations:

```bibtex
@conference {rizzuti2020SEGuqavp,
	title = {Parameterizing uncertainty by deep invertible networks, an application to reservoir characterization},
	booktitle = {SEG Technical Program Expanded Abstracts},
	year = {2020},
	note = {Accepted in SEG},
	month = {4},
	abstract = {Uncertainty quantification for full-waveform inversion provides a probabilistic characterization of the ill-conditioning of the problem, comprising the sensitivity of the solution with respect to the starting model and data noise. This analysis allows to assess the confidence in the candidate solution and how it is reflected in the tasks that are typically performed after imaging (e.g., stratigraphic segmentation following reservoir characterization). Classically, uncertainty comes in the form of a probability distribution formulated from Bayesian principles, from which we seek to obtain samples. A popular solution involves Monte Carlo sampling. Here, we propose instead an approach characterized by training a deep network that "pushes forward" Gaussian random inputs into the model space (representing, for example, density or velocity) as if they were sampled from the actual posterior distribution. Such network is designed to solve a variational optimization problem based on the Kullback-Leibler divergence between the posterior and the network output distributions. This work is fundamentally rooted in recent developments for invertible networks. Special invertible architectures, besides being computational advantageous with respect to traditional networks, do also enable analytic computation of the output density function. Therefore, after training, these networks can be readily used as a new prior for a related inversion problem. This stands in stark contrast with Monte-Carlo methods, which only produce samples. We validate these ideas with an application to angle-versus-ray parameter analysis for reservoir characterization.},
	keywords = {Full-waveform inversion, SEG, Uncertainty quantification},
	url = {https://slim.gatech.edu/Publications/Public/Conferences/SEG/2020/rizzuti2020SEGuqavp/rizzuti2020SEGuqavp.html},
	author = {Gabrio Rizzuti and Ali Siahkoohi and Philipp A. Witte and Felix J. Herrmann}
}
```

## Contact

Gabrio Rizzuti: rizzuti.gabrio@gatech.edu
