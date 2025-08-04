# -*- coding: utf-8 -*-

import time, math, threading, requests, pickle, json
from flask import Flask, Response
import numpy as np
import statsmodels.api as sm

# Global Config
TOKENS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
API_BASE = "https://www.okx.com"
MODEL_DIR = "models"
UPDATE_INTERVAL = 300  # 5 mins
DATA_POINTS = 180

app = Flask(__name__)

import os
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def getData(token):
    params = {
        "instId": token,
        "bar": "1D",
        "limit": str(min(DATA_POINTS, 300))
    }
    resp = requests.get(f"{API_BASE}/api/v5/market/candles", params=params, timeout=10)
    resp.raise_for_status()
    resp_json = resp.json()
    if resp_json.get("code") != "0":
        raise RuntimeError(f"OKX Error: {resp_json.get('msg')}")
    data = resp_json["data"]
    result = sorted([(int(item[0]) // 1000, float(item[4])) for item in data], key=lambda x: x[0])
    return result

def compute_log_returns(series):
    n = len(series)
    if n < 2:
        raise ValueError("Too few data points for log-return")
    prices = np.array([p for (_, p) in series], dtype=float)
    timestamps = np.array([t for (t, _) in series], dtype=float)
    log_returns = np.log(prices[1:] / prices[:-1])
    X = np.column_stack((prices[:-1], timestamps[:-1]))
    y = log_returns
    return X, y

def train_model(token):
    series = getData(token)
    X, y = compute_log_returns(series)
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    with open(f"{MODEL_DIR}/{token}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model for {token} trained and saved.")

def update_task():
    for token in TOKENS:
        try:
            train_model(token)
        except Exception as e:
            print(f"[ERROR] Failed to update {token} model:", e)

def update_loop():
    while True:
        try:
            update_task()
        except Exception as e:
            print("update_loop error:", e)
        time.sleep(UPDATE_INTERVAL)

@app.route("/inference/<string:token>")
def generate_inference(token):
    token = token.upper()
    if token not in TOKENS:
        return Response(json.dumps({"error": f"Token {token} not supported"}), status=400, mimetype="application/json")

    model_path = f"{MODEL_DIR}/{token}.pkl"
    if not os.path.exists(model_path):
        return Response(json.dumps({"error": f"Model for {token} not trained yet"}), status=500, mimetype="application/json")

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return Response(json.dumps({"error": "Failed to load model", "detail": str(e)}), status=500, mimetype="application/json")

    try:
        series = getData(token)
        if len(series) < 2:
            return Response(json.dumps({"error": "Not enough data for inference"}), status=500, mimetype="application/json")

        ts_current, price_current = series[-2]
        ts_next, price_next = series[-1]

        X_pred = np.array([[price_current, ts_current]])
        X_pred_with_const = sm.add_constant(X_pred, has_constant="add")
        pred_log_return = float(model.predict(X_pred_with_const)[0])
        predicted_price = price_current * math.exp(pred_log_return)

        result = {
            "token": token,
            "ts_current": ts_current,
            "price_current": price_current,
            "pred_log_return": pred_log_return,
            "predicted_price": predicted_price,
            "actual_price_next_day": price_next
        }
        return Response(str(pred_log_return), status=200, mimetype="application/json")

    except Exception as e:
        return Response(json.dumps({"error": "Inference error", "detail": str(e)}), status=500, mimetype="application/json")

@app.route("/update")
def http_update():
    try:
        update_task()
        return "0"
    except Exception as e:
        print("Manual update error:", e)
        return "1"

if __name__ == "__main__":
    update_task()
    thread = threading.Thread(target=update_loop, daemon=True)
    thread.start()
    app.run(host="0.0.0.0", port=8000)
