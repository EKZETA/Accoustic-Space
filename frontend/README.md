# AcousticSpace dashboard

## Run locally

From the repository root, start the API in one terminal:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

Then start the dashboard in another terminal:

```powershell
cd frontend
npm run dev
```

The dashboard uses `http://localhost:8000` as its API by default. Set
`VITE_API_URL` in a local frontend `.env` file to use a different API origin.

The Week 1 API extracts a compact mel spectrogram and descriptive acoustic
features plus a compact time-domain waveform envelope. The waveform shows the
minimum and maximum sample amplitude per time bin for the dashboard; it is not
a playable audio stream. The API does not classify audio as real or synthetic.
The decay value is only an exploratory energy-envelope proxy, not a measured
room impulse response.

The Week 2 dashboard renders the returned waveform alongside the mel
spectrogram after audio feature extraction.

Run `npm run lint` and `npm run build` to check the frontend.
