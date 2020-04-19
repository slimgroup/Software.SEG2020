# Transfer learning in large-scale ocean bottom seismic wavefield reconstruction

Codes for generating results in Zhang, M., Siahkoohi, A., and Herrmann, F.J., Transfer learning in large-scale ocean bottom seismic wavefield reconstruction. [arXiv:2004.07388](https://arxiv.org/abs/2004.07388).

## Prerequisites

This code has been tested using Deep Learning AMI (Amazon Linux) Version 24.2 on Amazon Web Services (AWS). We performed the test on `g4dn.xlarge` instances. Follow the steps below to install the necessary libraries:

```bash
cd $HOME
git clone https://github.com/slimgroup/Software.SEG2020.git
cd $HOME/Software.SEG2020/tree/master/zhang2020SEGtli
conda create -n wavefield-reconstruction pip python=3.6
source activate wavefield-reconstruction
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch #If your system has GPU
pip install --user -r  requirements.txt

```

## Dataset

Links have been provided in `RunTraining.sh` and `RunTransferLearning.sh` script to automatically download the ``14.33`` and ``14.66`` Hz monochromatic seismic data, respectively, into the necessary directory. Total size of the dataset for each fequency is 6.52GB + 6.52GB + 6.52GB + 118KB.

## Script descriptions

`RunTraining.sh`\: script for running training. It will make `model/` and `data/` directory in `$HOME` for storing training/testing data and saved neural net checkpoints and final results, respectively. Next, it will train a neural net for the experiment for ``14.33`` Hz monochromatic seismic data.

`RunTesting.sh`\: script for testing the trained neural net. It will reconstruct the entire subsampled ``14.33`` Hz monochromatic seismic data and place the result in `sample/` directory to be used for plotting purposes.

`RunTransferLearning.sh`\: script for running transfer learning on CNN trained to recover ``14.33`` Hz data in order to recover ``14.66`` Hz data. It will load the pre-trained neural net and perform transfer learning.

`src/main.py`\: constructs `wavefield_reconstrcution` class using given arguments in `RunTraining.sh`\, defined in `model.py` and calls `train` function in the defined  `wavefield_reconstrcution` class.

`src/model.py`: includes `wavefield_reconstrcution` class definition, which involves `train` and `test` functions.

### Running the code

To perform training, run:

```bash
# Running on GPU

bash RunTraining.sh

```

To perform transfer learning, after pre-training, run:

```bash
# Running in GPU

bash RunTransferLearning.sh

```

To evaluated the trained network on test data set (``14.66`` Hz) run the following. It will automatically load the latest checkpoint saved.

```bash
# Running on GPU

bash RunTesting.sh

```

To generate and save figures shown in paper for ``14.66`` Hz monochromatic seismic data run the following:

```bash

bash utilities/genFigures.sh

```

The saving directory can be changed by modifying `savePath` variable in `utilities/genFigures.sh`\.

## Citation

If you find this software useful in your research, please cite:

```bibtex
@unpublished {zhang2020SEGtli,
	title = {Transfer learning in large-scale ocean bottom seismic wavefield reconstruction},
	year = {2020},
	note = {Submitted to SEG},
	month = {4},
	author = {Mi Zhang and Ali Siahkoohi and Felix J. Herrmann},
	journal={arXiv preprint arXiv:2004.07388},
	url = {https://arxiv.org/pdf/2004.07388.pdf}
}
```

## Questions

Please contact alisk@gatech.edu for further questions.


## Author

Ali Siahkoohi

