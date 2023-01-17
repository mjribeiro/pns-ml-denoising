# pns-ai-denoising

Code for comparing two ML models (VAE and Noise2Noise) with conventional bandpass filtering. 

## Relevant files:
```
pns-ai-denoising
│   README.md
│   generate_mov_rms_figure.py
|   preprocess.py
|   train_n2n.py
|   train_vae.py    
│
└───datasets
│   │   vagus_dataset.py
│   
└───models
|   │   noise2noise_model.py
|   │   vae.py
|
└───utils
|   |   preprocessing.py
|   
```

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

Alternatively, steps 4-6 can also be replaced by just using pretrained/previously obtained results (just ask me for those if needed)