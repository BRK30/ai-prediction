from flask import Flask, jsonify, send_from_directory
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from prediction_model import predict_stock
from sector_stocks import ALL_STOCKS

app = Flask(__name__)

@app.get("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.get("/api/health")
def health():
    return jsonify({"success": True, "service": "AI Prediction Model"})

@app.get("/api/prediction/stocks")
def prediction_stocks():
    return jsonify({"success": True, "count": len(ALL_STOCKS), "stocks": ALL_STOCKS})

@app.get("/api/prediction/<symbol>")
def get_prediction(symbol):
    try:
        symbol = symbol.upper().strip()
        if symbol not in ALL_STOCKS and not symbol.endswith((".NS", ".BO")):
            return jsonify({"success": False, "message": f"{symbol} is not in the configured stock list."}), 404
        prediction = predict_stock(symbol)
        return jsonify({"success": True, "symbol": symbol, **prediction})
    except Exception as exc:
        app.logger.exception("Prediction failed for %s", symbol)
        return jsonify({"success": False, "message": str(exc)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
