import json
from flask import Flask, Response
import ccxt
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta
import os
import pickle
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler

# 配置参数
TOKENS_CONFIG = [
    {'symbol': 'BERA/USDT', 'timeframe': '1h', 'model_path': 'ols_model_BERA_1h.pkl', 'data_path': 'BERA_USDT_1h_data.csv'}
]

# 初始化 OKX
okx = ccxt.okx()

def download_data(symbol, timeframe, limit=168):
    """从 OKX 下载历史数据（支持指定数量）"""
    try:
        if limit is not None:
            # 直接获取最近的limit条数据
            ohlcv = okx.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        else:
            # 原时间范围逻辑（其他币种保持14天）
            end_time = datetime.now()
            print(end_time)
            start_time = end_time - timedelta(hours=168)
            print(start_time)
            ohlcv = okx.fetch_ohlcv(
                symbol, 
                timeframe=timeframe, 
                since=okx.parse8601(start_time.isoformat())
            )
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return pd.DataFrame()

    if len(ohlcv) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    return df

def train_ols_model(symbol, timeframe, data_path, model_path):
    """训练 OLS 模型"""
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            return
    except FileNotFoundError:
        return

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # 准备特征和目标变量
    df['target'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    
    if len(df) < 10:  # 确保有足够的数据
        return
    
    X = df[['open', 'high', 'low', 'close']]
    X = sm.add_constant(X)
    y = df['target']
    
    try:
        model = sm.OLS(y, X)
        results = model.fit()
        with open(model_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Model updated for {symbol} {timeframe}")
    except Exception as e:
        print(f"Error training model for {symbol}: {e}")

def update_data():
    """更新所有数据并训练模型"""
    print("\nStarting data update...")
    for config in TOKENS_CONFIG:
        symbol = config['symbol']
        timeframe = config['timeframe']
        data_path = config['data_path']
        model_path = config['model_path']
        #limit = config.get('limit')  # 获取limit参数
        
        # 下载数据（传入limit参数）
        df = download_data(symbol, timeframe)
        if not df.empty:
            df.to_csv(data_path)
            print(f"Data updated for {symbol} {timeframe}")
            
            # 训练模型
            train_ols_model(symbol, timeframe, data_path, model_path)
    print("Data update completed.\n")

def create_app():
    """创建 Flask 应用"""
    app = Flask(__name__)

    @app.route("/predict_log_return/<string:token>")
    def predict_price(token):
        """生成预测结果"""
        token = token.upper()
        config = None
        for cfg in TOKENS_CONFIG:
            if cfg['symbol'].startswith(token + "/"):
                config = cfg
                break
        
        if not config:
            return Response(json.dumps({"error": "Token not supported"}), 400)
        
        try:
            # 加载模型
            with open(config['model_path'], "rb") as f:
                model = pickle.load(f)
            
            # 读取最新数据
            df = pd.read_csv(config['data_path'])
            if df.empty:
                return Response(json.dumps({"error": "No data available"}), 500)
            
            # 准备预测数据
            latest = df.iloc[-1]
            #print(df.tail(1)['close'])
            X_new = pd.DataFrame([[latest['open'], latest['high'], latest['low'], latest['close']]],
                                columns=['open', 'high', 'low', 'close'])
            X_new = sm.add_constant(X_new, has_constant='add')
            
            # 进行预测
            prediction = model.predict(X_new)[0]
            log_return = np.log(prediction/(df.tail(1)['close'].values[0]))
            print(prediction)
            print(df.tail(1)['close'].values[0])
            return Response(f"{log_return}", mimetype='text/plain')
        
        except Exception as e:
            return Response(json.dumps({"error": str(e)}), 500)

    @app.route("/update")
    def update():
        """更新数据并返回状态"""
        try:
            update_data()
            return "0"
        except Exception as e:
            print(f"Error during update: {e}")
            return "1"

    return app

def scheduled_update():
    """定时任务更新数据"""
    with app.app_context():
        try:
            update_data()
        except Exception as e:
            print(f"Scheduled update error: {e}")

if __name__ == "__main__":
    # 初始化应用
    app = create_app()
    
    # 首次运行更新数据
    if not all(os.path.exists(c['model_path']) for c in TOKENS_CONFIG):
        print("Initial data download...")
        update_data()

    # 配置定时任务
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_update, 'interval', minutes=1)
    scheduler.start()
    
    # 启动Flask应用
    app.run(host="0.0.0.0", port=8000)
