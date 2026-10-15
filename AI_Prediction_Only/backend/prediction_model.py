"""Per-stock LSTM forecasting pipeline for MarketMentor.

Downloads real daily prices from Yahoo Finance through yfinance. No bundled or
fabricated price records are used. A separate cached Keras model is maintained
for each resolved Yahoo Finance ticker. Forecasts are experimental, not advice.
"""
from pathlib import Path
import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential, load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
WINDOW = 60
EPOCHS = 20
BATCH_SIZE = 32
MIN_HISTORY = WINDOW + 100


def _safe_symbol(symbol):
    symbol = str(symbol).upper().strip()
    if not re.fullmatch(r"[A-Z0-9&._-]{1,24}", symbol):
        raise ValueError("Invalid stock symbol")
    return symbol


def _fetch_history(symbol):
    """Try the supplied Yahoo ticker, then NSE and BSE suffixes."""
    raw = _safe_symbol(symbol)
    candidates = [raw] if raw.endswith((".NS", ".BO")) else [raw + ".NS", raw + ".BO"]
    last_error = None
    for yahoo_symbol in dict.fromkeys(candidates):
        try:
            data = yf.Ticker(yahoo_symbol).history(
                period="5y", interval="1d", auto_adjust=True, actions=False
            )
            if data is not None and not data.empty and "Close" in data.columns:
                close = data[["Close"]].copy()
                close["Close"] = pd.to_numeric(close["Close"], errors="coerce")
                close = close.replace([np.inf, -np.inf], np.nan).dropna()
                close = close[~close.index.duplicated(keep="last")].sort_index()
                if len(close) >= MIN_HISTORY:
                    return yahoo_symbol, close
                if len(close):
                    last_error = RuntimeError(
                        f"Only {len(close)} daily prices are available for {raw}; "
                        f"at least {MIN_HISTORY} are required for this LSTM."
                    )
        except Exception as exc:
            last_error = exc
    if last_error:
        raise RuntimeError(f"Unable to fetch enough real historical prices for {raw}: {last_error}")
    raise RuntimeError(f"No historical NSE/BSE price data found for {raw}.")


def _paths(yahoo_symbol):
    safe = re.sub(r"[^A-Z0-9_-]", "_", yahoo_symbol.upper())
    return (
        MODEL_DIR / f"{safe}.keras",
        MODEL_DIR / f"{safe}_scaler.pkl",
        MODEL_DIR / f"{safe}_meta.json",
    )


def _sequences(scaled_values):
    values = np.asarray(scaled_values, dtype=np.float32).reshape(-1)
    X = np.stack([values[i - WINDOW:i] for i in range(WINDOW, len(values))])
    y = values[WINDOW:]
    return X.reshape(-1, WINDOW, 1), y


def _new_model():
    model = Sequential([
        Input(shape=(WINDOW, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model


def _direction(predicted, current, deadband=0.001):
    change = (predicted - current) / current if current else 0.0
    if change > deadband:
        return "UP"
    if change < -deadband:
        return "DOWN"
    return "NEUTRAL"


def _train_and_evaluate(prices):
    """Evaluate chronologically, then train the deployable model on all history."""
    split_at = int(len(prices) * 0.8)
    if split_at <= WINDOW or len(prices) - split_at < 20:
        raise RuntimeError("Not enough data for a chronological train/test split.")

    # Evaluation scaler sees only the training partition.
    eval_scaler = MinMaxScaler(feature_range=(0, 1))
    eval_scaler.fit(prices[:split_at])
    scaled_eval = eval_scaler.transform(prices)
    X, y = _sequences(scaled_eval)
    target_indices = np.arange(WINDOW, len(prices))
    train_mask = target_indices < split_at
    test_mask = target_indices >= split_at
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    if len(X_train) < 50 or len(X_test) < 20:
        raise RuntimeError("Not enough training or holdout sequences for reliable evaluation.")

    evaluation_model = _new_model()
    stopper = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    history = evaluation_model.fit(
        X_train, y_train, validation_split=0.15, epochs=EPOCHS,
        batch_size=BATCH_SIZE, shuffle=False, callbacks=[stopper], verbose=0,
    )
    pred_scaled = evaluation_model.predict(X_test, verbose=0)
    pred_prices = eval_scaler.inverse_transform(pred_scaled).ravel()
    actual_prices = eval_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    previous_prices = prices[target_indices[test_mask] - 1, 0]
    mae = float(mean_absolute_error(actual_prices, pred_prices))
    rmse = float(np.sqrt(mean_squared_error(actual_prices, pred_prices)))
    direction_accuracy = float(np.mean(
        (_direction_vector(actual_prices, previous_prices)) ==
        (_direction_vector(pred_prices, previous_prices))
    ) * 100.0)

    # Fit a fresh production model on all available history after evaluation.
    final_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_all = final_scaler.fit_transform(prices)
    X_all, y_all = _sequences(scaled_all)
    final_model = _new_model()
    final_stopper = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    final_history = final_model.fit(
        X_all, y_all, epochs=min(EPOCHS, max(8, len(history.history.get("loss", [])))),
        batch_size=BATCH_SIZE, shuffle=False, callbacks=[final_stopper], verbose=0,
    )
    return final_model, final_scaler, {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": direction_accuracy,
        "training_samples": int(len(X_all)),
        "test_samples": int(len(X_test)),
        "epochs_completed": int(len(final_history.history.get("loss", []))),
        "evaluation_epochs": int(len(history.history.get("loss", []))),
    }


def _direction_vector(values, previous):
    values = np.asarray(values)
    previous = np.asarray(previous)
    change = np.divide(values - previous, previous, out=np.zeros_like(values, dtype=float), where=previous != 0)
    return np.where(change > 0.001, 1, np.where(change < -0.001, -1, 0))


def predict_stock(symbol):
    requested_symbol = _safe_symbol(symbol)
    yahoo_symbol, data = _fetch_history(requested_symbol)
    prices = data["Close"].astype("float32").to_numpy().reshape(-1, 1)
    last_date = str(data.index[-1].date())
    model_path, scaler_path, meta_path = _paths(yahoo_symbol)

    model = scaler = None
    meta = {}
    if model_path.exists() and scaler_path.exists() and meta_path.exists():
        try:
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if cached_meta.get("last_training_date") == last_date:
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                meta = cached_meta
        except Exception as exc:
            warnings.warn(f"Cached LSTM could not be loaded; retraining {yahoo_symbol}: {exc}")

    if model is None or scaler is None:
        model, scaler, metrics = _train_and_evaluate(prices)
        meta = {
            "symbol": requested_symbol,
            "yahoo_symbol": yahoo_symbol,
            "model_type": "Stacked LSTM (64 + 32 units)",
            "last_training_date": last_date,
            "window_days": WINDOW,
            **metrics,
        }
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    current_price = float(prices[-1, 0])
    scaled_window = scaler.transform(prices[-WINDOW:]).reshape(1, WINDOW, 1)
    rolling = scaled_window.copy()
    forecasts = []
    for _ in range(7):
        next_scaled = float(model.predict(rolling, verbose=0)[0, 0])
        next_price = float(scaler.inverse_transform(np.array([[next_scaled]], dtype=np.float32))[0, 0])
        forecasts.append(max(0.0, next_price))
        rolling = np.concatenate(
            [rolling[:, 1:, :], np.array(next_scaled, dtype=np.float32).reshape(1, 1, 1)], axis=1
        )

    tomorrow = forecasts[0]
    tomorrow_change_pct = ((tomorrow - current_price) / current_price * 100.0) if current_price else 0.0
    # A descriptive score, not a probability of profit or calibrated confidence.
    error_pct = (float(meta.get("mae", 0.0)) / current_price * 100.0) if current_price else 100.0
    model_score = int(round(max(0.0, min(100.0, float(meta.get("directional_accuracy", 0.0)) * 0.7 + max(0.0, 100.0 - error_pct * 10.0) * 0.3))))
    return {
        "current_price": round(current_price, 2),
        "tomorrow": round(tomorrow, 2),
        "week": round(forecasts[-1], 2),
        "direction": _direction(tomorrow, current_price),
        "trend": "Bullish" if forecasts[-1] > current_price else "Bearish" if forecasts[-1] < current_price else "Neutral",
        "expected_change_percent": round(tomorrow_change_pct, 2),
        "model_score": model_score,
        "confidence": model_score,  # legacy UI compatibility; not a probability of profit
        "mae": round(float(meta.get("mae", 0.0)), 4),
        "rmse": round(float(meta.get("rmse", 0.0)), 4),
        "directional_accuracy": round(float(meta.get("directional_accuracy", 0.0)), 2),
        "model_type": meta.get("model_type", "Stacked LSTM"),
        "training_date": last_date,
        "training_samples": int(meta.get("training_samples", 0)),
        "test_samples": int(meta.get("test_samples", 0)),
        "epochs_completed": int(meta.get("epochs_completed", 0)),
        "forecast_horizon_days": 7,
        "yahoo_symbol": yahoo_symbol,
        "disclaimer": "Experimental historical-price forecast; the model score is not a probability of profit or investment advice.",
    }
