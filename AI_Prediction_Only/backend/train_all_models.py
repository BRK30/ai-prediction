"""Train/cache an individual LSTM for every symbol in the sector directory.

Run from the backend folder with the project's virtual environment active:
    python train_all_models.py
This downloads real historical data on demand. It can take a long time and may
skip symbols whose history is unavailable or insufficient. No prices are bundled.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from prediction_model import predict_stock
from sector_stocks import ALL_STOCKS

BASE_DIR = Path(__file__).resolve().parent
REPORT_FILE = BASE_DIR / "training_report.json"


def main():
    results = []
    total = len(ALL_STOCKS)
    for index, symbol in enumerate(ALL_STOCKS, start=1):
        print(f"[{index}/{total}] Training/checking {symbol}...", flush=True)
        started = time.time()
        try:
            prediction = predict_stock(symbol)
            results.append({
                "symbol": symbol,
                "status": "ok",
                "training_date": prediction.get("training_date"),
                "directional_accuracy": prediction.get("directional_accuracy"),
                "mae": prediction.get("mae"),
                "rmse": prediction.get("rmse"),
                "seconds": round(time.time() - started, 1),
            })
            print(
                f"  OK | {prediction.get('yahoo_symbol')} | "
                f"holdout directional accuracy: {prediction.get('directional_accuracy')}%",
                flush=True,
            )
        except Exception as exc:
            results.append({
                "symbol": symbol,
                "status": "skipped",
                "error": str(exc),
                "seconds": round(time.time() - started, 1),
            })
            print(f"  SKIPPED | {exc}", flush=True)
        REPORT_FILE.write_text(json.dumps({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_symbols": total,
            "completed": sum(item["status"] == "ok" for item in results),
            "skipped": sum(item["status"] != "ok" for item in results),
            "results": results,
        }, indent=2), encoding="utf-8")
    print(f"\nFinished. Report: {REPORT_FILE}", flush=True)


if __name__ == "__main__":
    main()
