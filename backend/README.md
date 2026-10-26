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

The current endpoint reports acoustic descriptors only; it does not perform a
deepfake classification.

Run the backend unit tests from the `backend` directory:

```powershell
python -m unittest discover -s tests
```
