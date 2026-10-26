# AcousticSpace backend

## Run the API

Create and activate a virtual environment if one is not already available, then
install the requirements:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`; interactive API documentation
is at `/docs`. The feature endpoint accepts WAV, MP3, and FLAC uploads up to
25 MB at `/api/v1/extract-features`.

## Prepare the ASVspoof 2019 LA manifest

Place the extracted LA dataset at `backend/data/asvspoof/LA/LA`, preserving the
official protocol and split directory names. Then run from the repository root:

```powershell
python backend/scripts/prepare_asvspoof_manifest.py
```

This validates the official train, development, and evaluation protocols against
their FLAC files, and writes a labeled CSV to
`backend/data/manifests/asvspoof2019_la.csv`. The audio corpus is not copied or
modified by the script. Keep the dataset's supplied license and citation with
any research use.

The RIRS_NOISES corpus is stored separately under `backend/data/rir`. Its
impulse responses are useful for later controlled reverberation experiments;
they are not treated as ground-truth room responses for arbitrary uploaded
speech recordings.

## Extract batch features with an RIR

After creating the ASVspoof manifest, run from the repository root:

```powershell
python backend/scripts/prepare_data.py
```

The script applies the selected RIR to each audio file in the manifest, then
writes one row per processed recording to
`backend/data/rir_extracted_features.csv`. Rows include the mean spectral
centroid, mean spectral rolloff, and the 64-by-128 mel spectrogram as compact
JSON in `spectrogram_values_db`. Failed recordings are reported and skipped;
the command exits with a non-zero status if any entry fails.

Use `--split train`, `--split dev`, or `--split eval` to select a split, and
`--limit 100` for a bounded test run. Paths can be overridden with
`--manifest`, `--dataset-root`, `--rir`, and `--output`. For example:

```powershell
python backend/scripts/prepare_data.py --split train --limit 100
```

## Train the Week 2 CNN baseline

Install the updated backend requirements, generate the feature CSV, then train
from the repository root:

```powershell
python -m pip install -r backend/requirements.txt
python backend/scripts/train_baseline_cnn.py --epochs 15
```

The baseline trains on the `train` split and selects its checkpoint using the
`dev` split. It does not use the `eval` split. The best model is saved to
`backend/data/models/rir_baseline_cnn.pt`, with validation metrics in the
adjacent `.metrics.json` file. Metrics include accuracy and spoof precision,
recall, and F1. Tune training with `--batch-size`, `--learning-rate`, and
`--seed`.

This is an experimental baseline, not a validated deepfake detector. The
current API still reports acoustic features only and does not serve model
predictions.

Run the backend unit tests from the `backend` directory:

```powershell
python -m unittest discover -s tests
```
