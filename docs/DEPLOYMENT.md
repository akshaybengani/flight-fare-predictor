# Deploying FareSight

The app is a small Flask service (`src/app.py`, object name `app`). It needs the merged data table and the trained model at start-up; neither is stored in git (data is third-party, the model file is about 110 MB).

## 1. Install
```bash
git clone <this repo> && cd flight-fare-predictor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Get the model and the data table
Pick one:
- **Copy them** from the machine that trained them: `data/merged.parquet` and `models/fare_model.joblib` (keep the same relative paths). Fastest.
- **Rebuild them**: `bash scripts/download_data.sh` (needs your own Kaggle token in `~/.kaggle/`), then `python src/build_dataset.py` and `python src/train.py --final-only` (about 10 minutes on a laptop CPU).

## 3. Run
Development: `cd src && python app.py` (port 5055, override with `PORT`).

Production:
```bash
cd src && gunicorn -w 2 -b 127.0.0.1:5055 --timeout 60 app:app
```
Each worker loads the 110 MB model, so budget about 1 GB of RAM per worker. Put it behind a reverse proxy (nginx or similar) that adds HTTPS.

## 4. Things to know before exposing it
- **No authentication.** The Model & Insights page can change prediction settings (`POST /api/config`). Restrict it at the proxy, or keep the service on an internal network.
- **Config** is written to `config.json` next to the app. Give the service user write access to that folder, or mount it as a volume.
- **Templates are cached** when debug is off, so restart workers after editing `templates/`.
- **Licences of data.** The code is MIT. Open-Meteo's free API is non-commercial only and one of the downloaded fare sets (Goibibo) is CC BY-NC-SA, so check each source's terms before commercial use. See `docs/ENRICHED_FEATURES.md` and `docs/DATA_SCOUT_REPORT.md`.
- Predictions are model estimates from historical scraped fares, not booking advice.
