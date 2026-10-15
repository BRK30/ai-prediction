# AI Prediction Model (prediction-only)

This trimmed project contains only the stock prediction interface, Flask prediction API, per-stock stacked LSTM pipeline, sector stock symbol list, and optional sequential training script. It does not include the dashboard, live-market browser, stock comparison page, portfolio, or authentication pages.

## Windows setup (Python 3.11, PowerShell)

From this project folder, activate your existing environment if it is at `D:\AI_Prediction_Env`:

```powershell
D:\AI_Prediction_Env\Scripts\Activate.ps1
$env:TEMP = "D:\AI_Temp"
$env:TMP = "D:\AI_Temp"
$env:PIP_CACHE_DIR = "D:\AI_Pip_Cache"
python -m pip install --no-cache-dir -r .\backend\requirements.txt
python .\backend\app.py
```

Open http://127.0.0.1:5000/ . API endpoints: `/api/health`, `/api/prediction/stocks`, and `/api/prediction/TCS`.

## Model behavior

No fabricated price history or pretrained weights are bundled. On the first prediction for a symbol, the model fetches up to five years of daily adjusted close data from Yahoo Finance, trains/evaluates chronologically, and caches its model/scaler/metrics under `backend/trained_models/`. Models are reused while the latest available market date is unchanged. Some symbols may fail if Yahoo Finance has no suitable ticker/history. Predictions are experimental and not investment advice.

## Optional batch training

From the project root:

```powershell
python .\backend\train_all_models.py
```

This can take hours and depends on network/data availability.
