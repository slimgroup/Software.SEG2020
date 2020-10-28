

## Data

 The ground truth data collected from the Gulf of Suez [Full.mat](https://slim.gatech.edu/PublicationsData/zhang2020SEGwrw/). It should be downloaded and saved in *data/exp_raw/*

*data/exp_raw/*:
 
 This directory contains Jittered subsampling indexes[ind.mat] and one unweighted initial prior information[Idx_30.mat].
 
 
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


