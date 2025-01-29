# MODIFIED AI-music detection study
MODIFIED Code repository of our research paper on AI-generated music detection - D. Afchar, G. Meseguer Brocal, R. Hennequin (2024).

The modification concerns adaptations to data format and augmentations to conform with other choices we made.

## Setup
docker build -t robust_det .

Note that this needs a separate (TF-based) image.

### VS Code connect
rocker -netapp -gpu -vault -ti --rm --ipc=host --memory=128g --name=USER_aigm-det_container --entrypoint bash -v /data/nfs/analysis/USER/deepfake-detector/deepfake-detector/:/workspace/deepfake-detector aigm-det

For further options and info, see the main robust-detection repo.

## Data
We need songs in local directories. To get this, use `data_utils/copy_real.py` and `data_utils/copy_fake.py`.


## Training
Run `scripts/train.py`; it will take care of everything. For text generation model stratification experiments, run with `--data_setup stratfify_X`, (X: 1,2,3).

## Evaluation
Run `scripts/eval.py`. With no arguments for the default (in-domain) scenario. For text stratification experiments, run with `--data_setup stratify_X`. For audio attacks/Udio, run with `--adversarial Y`, Y being one of ["udio", "pitch", "reverb", "eq", "noise", "stretch"]

Finally, we use these predictions to get a full predictions output file that resembles the `Detector` class used in the other experiments, done with `data_utils/get_preds_file.py`, and separately for each run: base, audio OOD: ["udio", "pitch", "reverb", "eq", "noise", "stretch"], and text OOD: ["specnn_amplitude_base_mistral-tinyllama-train125", "specnn_amplitude_base_wizardlm2-mistral-train125", "specnn_amplitude_base_wizardlm2-tinyllama-train125"]
I then just copy them (saved to `predictions/`) into the `artifactsV3` path where all other predictions are saved. Then, they can be evaluated in the very same way as all other methods.