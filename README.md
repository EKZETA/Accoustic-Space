# AcousticSpace

AcousticSpace is an experimental audio-analysis dashboard and machine-learning
project for exploring acoustic features in bona fide and spoof recordings. It
provides an upload-and-analysis workflow, dataset preparation and RIR
experiments, a CNN baseline, and an optional Hugging Face Audio Spectrogram
Transformer (AST) training and inference path.

> **Important:** AcousticSpace is a research prototype, not a validated
> forensic tool. It does not isolate a room impulse response (RIR) from an
> arbitrary uploaded recording, detect breathing or syllables, or establish
> whether an audio file is genuine. AST predictions are experimental model
> outputs, not calibrated certainty.

## What the project does

### Analyst dashboard

- Accepts WAV, MP3, and FLAC files up to 25 MB through file selection or drag
  and drop.
- Plays the selected audio locally in the browser before analysis.
- Displays an amplitude-envelope waveform and a mel spectrogram returned by
  the API.
- Summarizes extracted signal indicators and, when an AST model is configured,
  displays its predicted label and class probabilities.
- Shows energy-based activity and quiet intervals on a timeline. These are
  timing proxies, not breathing or speech-segment detection.
- Keeps up to ten recent analysis results in this browser's local storage.
  History contains extracted results, not uploaded audio; entries can be
  reopened or deleted.

### Audio API and features

The FastAPI service decodes input as mono audio at 16 kHz and returns:

- **Metadata:** sample rate, duration, and sample count.
- **Waveform envelope:** minimum and maximum amplitude for up to 1,200 time
  bins. This is a visualization summary, not the playable audio.
- **Mel spectrogram:** a 64-by-128 dB feature map.
- **Acoustic indicators:** mean spectral centroid, spectral rolloff, RMS
  energy, and an exploratory energy-decay proxy.
- **Temporal indicators:** threshold-based activity and quiet intervals,
  activity ratio, and mean pause duration.
- **Optional AST prediction:** returned only if a compatible local AST model is
  configured for the API.

Uploads are written to a temporary file while being processed and removed
afterward. The API does not keep the original uploaded audio.

### Dataset and machine-learning tools

- **ASVspoof manifest preparation:** validates official LA protocol entries
  against the extracted audio files and creates labeled CSV manifests.
- **Balanced sampling:** creates reproducible, class-balanced train, dev, and
  eval manifests.
- **RIR feature preparation:** convolves selected recordings with a known RIR
  and exports their features for controlled experiments.
- **RIR validation:** compares dry and reverberated versions of the same
  recordings and evaluates room-group recognition with source- and
  room-disjoint splits.
- **CNN baseline:** trains on extracted mel spectrograms, selects a checkpoint
  using dev loss, and evaluates it on the held-out eval split. This baseline is
  not currently served by the upload API.
- **Optional AST path:** fine-tunes a Hugging Face AST checkpoint on ASVspoof
  audio and can serve predictions when the saved model directory is configured.

RIR experiments measure controlled changes from known simulated room
responses. They do not prove that the system can infer the real room or isolate
an RIR from arbitrary speech.

## End-to-end analysis flow

1. The analyst selects or drops an audio file in the React dashboard.
2. The browser checks the extension and the 25 MB size limit, then creates a
   temporary browser URL for audio preview.
3. When **Extract features** is selected, the dashboard sends the file as
   multipart form data to `POST /api/v1/extract-features`.
4. The API validates the extension and size, writes a temporary upload, and
   decodes the recording to mono 16 kHz audio.
5. The audio processor builds the spectrogram, waveform envelope, acoustic
   indicators, and energy-based temporal intervals. If `ACOUSTICSPACE_AST_MODEL_DIR`
   points to a compatible local checkpoint, the optional AST classifier also
   predicts a label.
6. The API returns JSON. The dashboard renders the result and stores the
   extracted result in browser history. The temporary server-side upload is
   removed whether processing succeeds or fails.

## Machine-learning experiment flow

```text
ASVspoof audio + official protocols
                |
                v
       labeled CSV manifest
                |
                +----> balanced manifest (optional)
                |
                v
     feature extraction / RIR experiment
                |
                +----> CNN baseline on saved spectrogram features
                |
                +----> optional AST fine-tuning on audio
                              |
                              v
                    saved AST checkpoint
                              |
                              v
                 optional API inference
```

Train and dev data are used for fitting and checkpoint selection. The eval
split is held out for final evaluation and should not be used to tune model
choices. Keep the ASVspoof dataset license and citation with research use.

## Run locally

Prerequisites: Python 3.10 or newer and Node.js 22 or newer.

Start the backend in a PowerShell terminal from the repository root:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Start the dashboard in another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open the Vite URL printed in the terminal (normally
`http://localhost:5173`). The dashboard uses `http://localhost:8000` as its
default API URL. Set `VITE_API_URL` in `frontend/.env.local` if the API runs at
a different origin.

The API health check is at `http://localhost:8000/api/v1/health`; interactive
API documentation is at `http://localhost:8000/docs`.

## Run with Docker

Start Docker Desktop and wait for its engine to be running. From the repository
root:

```powershell
docker build -f backend\Dockerfile -t acousticspace-api .
docker run --rm -p 8000:8000 acousticspace-api
```

Check the API from another PowerShell terminal:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

The container runs the backend only; start the frontend separately as above.
AST inference is disabled unless an AST-enabled image is built and a saved
checkpoint is mounted and configured. See [backend/README.md](backend/README.md)
for Docker and model setup details.

## Prepare data and train models

The ASVspoof corpus and RIR corpus are not required to run the upload dashboard.
They are needed for the offline experiments and training workflows. Detailed
dataset paths, commands, and options are in
[backend/README.md](backend/README.md).

Main scripts:

| Script | Purpose |
| --- | --- |
| `backend/scripts/prepare_asvspoof_manifest.py` | Validate official protocols and write the labeled manifest. |
| `backend/scripts/sample_balanced_manifest.py` | Sample balanced labels per split for reproducible experiments. |
| `backend/scripts/prepare_data.py` | Apply an RIR and export acoustic and spectrogram features. |
| `backend/scripts/validate_rir_effects.py` | Measure paired dry/wet feature changes and room-group recognition. |
| `backend/scripts/train_baseline_cnn.py` | Train/evaluate the Week 2 CNN baseline from a feature CSV. |
| `backend/scripts/train_ast.py` | Fine-tune the optional Hugging Face AST directly from manifest audio. |

AST training dependencies are optional:

```powershell
python -m pip install -r backend/requirements-ast.txt
```

The default AST training command and model configuration are documented in
[backend/README.md](backend/README.md). Training downloads a pretrained model
and may require substantial disk space and a GPU. To enable inference, set
`ACOUSTICSPACE_AST_MODEL_DIR` to the saved AST directory before starting the API.

## Tests and continuous integration

Run backend tests from the `backend` directory:

```powershell
python -m unittest discover -s tests
```

Run frontend lint and production build from `frontend`:

```powershell
npm run lint
npm run build
```

The GitHub Actions workflow at `.github/workflows/ci.yml` runs backend tests,
frontend lint/build, and a backend Docker image build on pushes and pull
requests.

## Project layout

```text
backend/
  app/
    main.py                    FastAPI endpoints and temporary upload handling
    services/
      audio_processor.py       Audio features and temporal indicators
      ast_classifier.py        Optional AST inference
  scripts/                     Dataset, RIR, CNN, and AST workflows
  tests/                       Backend unit tests
frontend/
  src/
    App.tsx                    Dashboard layout and analysis state
    components/
      AudioUpload.tsx          File validation, preview, and API upload
      AnalysisResults.tsx      Model output and temporal timeline
      HistoryPanel.tsx         Local analysis history UI
    utils/
      analysisHistory.ts       Browser-local history storage
.github/workflows/ci.yml       Continuous-integration checks
```

## Current limitations and next validation

- The project does not yet implement breathing detection or alignment between
  breaths and spoken syllables.
- The decay value is an energy-envelope proxy, not a measured RIR.
- The temporal timeline uses an RMS energy threshold; silence, noise, and
  recording conditions can affect it.
- AST inference requires a trained, locally available model. The output
  probability is not calibrated confidence.
- The CNN baseline is experimental and is not connected to the upload API.
- RIR validation is controlled simulation, not real-world room recognition.
- Before treating any output as useful evidence, train/validate the AST
  workflow on the intended dataset, inspect held-out results, test API
  inference, and verify Docker/CI in an environment where Docker is running.

See [frontend/README.md](frontend/README.md) for dashboard-specific notes and
[backend/README.md](backend/README.md) for API, dataset, and model instructions.
