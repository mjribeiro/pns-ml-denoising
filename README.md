# Denoising and decoding spontaneous vagus nerve recordings with machine learning

This repository contains accompanying code for our IEEE Engineering in Medicine & Biology (EMBC) 2023 [paper](https://ieeexplore.ieee.org/abstract/document/10340443) on denoising and decoding spontaneous vagus nerve recordings with machine learning. More specifically, two ML models (VAE and Noise2Noise) were adapted, implemented, and compared with conventional bandpass filtering. Note that this repository is not actively maintained, and the related dataset is only available upon request. 

## Short file summaries:
* ``generate_mov_rms_figure.py``: Generates time-domain and moving RMS plots for draft paper, and prints some of the metrics. Requires models to have been run and inputs/reconstructions to have been saved. 
* ``preprocess.py``: Take the original `baselineFast.mat` file containing ENG data and generate train, validation, and test sets (currently labelled as "n2n" but being used for both VAE and N2N models)
* ``train_n2n.py``: Train Noise2Noise model
* ``train_vae.py``: Train VAE model
* ``datasets/vagus_dataset.py``: Take processed datasets and prepare them to be used in PyTorch (e.g. with DataLoader, batches, etc)
* ``models/noise2noise_model.py``: PyTorch implementation for Noise2Noise model.
* ``models/vae.py``: PyTorch implementation for VAE model.
* ``utils/preprocessing.py``: Useful functions for processing/splitting data.

## Running code
Code can be run as follows:

1. Install dependencies from `requirements.txt` (Windows) or `requirements.ubuntu22.txt` (Linux) in a venv, as follows:

    ```
    # Create new venv if needed
    python -m venv .venv
    pip install -r requirements.txt
    ```

2. Create the directories `data/Metcalfe-2014/`, `plots/`, and `results/` at the top level of the repo

3. Copy `baselineFast.mat` file into `data/Metcalfe-2014/`

4. Run `preprocess.py` to generate training, validation, and test sets.

5. Train VAE and Noise2Noise models using the corresponding training scripts and modifying any hyperparameters if needed (see file descriptions above)

6. Run `generate_mov_rms_figure.py` to generate plots and metrics from the manuscript.

Alternatively, steps 4-6 can also be replaced by just using pretrained/previously obtained results.
