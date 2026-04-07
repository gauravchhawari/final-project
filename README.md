# Flood Risk System

Real-time flood risk assessment and evacuation route optimization platform for low-lying urban regions.

## What This Project Now Includes

- FastAPI backend with separate weather, risk, and routing endpoints
- Streamlit dashboard for training, prediction, weather checks, route generation, and model evaluation
- Multi-model ML training pipeline for Random Forest, XGBoost, and LSTM
- Google Maps Directions and Elevation integration hooks for safe-route selection
- OpenWeather integration hooks for live weather and forecast data
- Shelter dataset support through `data/external/shelters.csv`

## Required Inputs

1. Historical flood dataset in `data/raw/` as `.csv` or `.parquet`
2. Shelter dataset at `data/external/shelters.csv`
3. API keys in `.env`

Copy `.env.example` to `.env` and fill in:

```env
OPENWEATHER_API_KEY=your_key_here
GOOGLE_MAPS_API_KEY=your_key_here
SHELTER_DATASET_PATH=data/external/shelters.csv
```

## Expected Training Columns

Your historical dataset should contain the target column you want to predict, plus as many of these as possible:

- `timestamp`
- `latitude`
- `longitude`
- `rainfall_mm_hr`
- `water_level_m`
- `temperature_c`
- `humidity_pct`
- `elevation_m`
- `distance_to_water_km`
- `drainage_capacity`
- `flood_risk_label`

## Shelter Dataset Schema

Use the header in `data/external/shelters_schema.csv` and save the real file as `data/external/shelters.csv`.

Required columns:

- `name`
- `latitude`
- `longitude`

Recommended columns:

- `region`
- `status`
- `capacity`
- `elevation_m`
- `address`

## Run The Backend

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend
```

## Run The Frontend

```powershell
.\venv\Scripts\python.exe -m streamlit run frontend\streamlit_app.py
```

## Train And Evaluate From CLI

```powershell
.\venv\Scripts\python.exe scripts\train_all_models.py
.\venv\Scripts\python.exe scripts\evaluate_models.py
```

## Important Note

The backend and dashboard start successfully now, but real risk prediction and real evacuation routes depend on:

- your actual Gurugram historical dataset
- your real shelter inventory
- valid OpenWeather and Google Maps API keys
 backend run command = python -m uvicorn app.main:app --reload --app-dir backend