"""
台積電(2330.TW)明日漲跌預測模型
==================================
流程:抓資料 -> 特徵工程 -> 定義標籤 -> Walk-forward 驗證 -> 訓練模型 -> 評估 -> 簡易回測

執行前請先安裝套件:
    pip install yfinance lightgbm scikit-learn pandas numpy matplotlib

執行:
    python tsmc_predict.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. 抓取歷史資料
# ----------------------------------------------------------------------
TICKER = "2330.TW"   # 台股上市台積電本股。美股ADR可改成 "TSM"
START = "2018-01-01"

def fetch_data(ticker=TICKER, start=START):
    df = yf.download(ticker, start=start, auto_adjust=True)

    if df is None or df.empty:
        raise ValueError(
            f"無法取得 {ticker} 的資料,請檢查:\n"
            f"  1. 網路連線是否正常\n"
            f"  2. 股票代碼是否正確(台股格式如 2330.TW)\n"
            f"  3. Yahoo Finance 服務是否暫時異常"
        )

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]  # 攤平多層欄位
    df = df.dropna()
    return df


# ----------------------------------------------------------------------
# 2. 特徵工程
#    每個特徵都要能講出金融邏輯,不是亂丟指標
# ----------------------------------------------------------------------
def add_features(df):
    df = df.copy()

    # 動量類:過去N日報酬率,捕捉趨勢延續性
    for n in [1, 3, 5, 10, 20]:
        df[f"return_{n}d"] = df["Close"].pct_change(n)

    # 波動率:過去N日報酬的標準差,反映市場不確定性
    df["volatility_10d"] = df["Close"].pct_change().rolling(10).std()
    df["volatility_20d"] = df["Close"].pct_change().rolling(20).std()

    # 均線乖離率:價格偏離均線的程度,常用於判斷超漲超跌
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma_gap"] = (df["ma5"] - df["ma20"]) / df["ma20"]

    # 成交量變化:量能是否異常放大,常伴隨轉折
    df["volume_change_5d"] = df["Volume"].pct_change(5)
    df["volume_ma_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # RSI(14日):判斷超買超賣
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # 當日振幅:反映當日市場情緒強度
    df["day_range"] = (df["High"] - df["Low"]) / df["Close"]

    return df


# ----------------------------------------------------------------------
# 3. 定義標籤:明日是否上漲(1=漲, 0=跌或平)
#    注意:標籤用「未來」報酬算出,訓練時絕對不能讓特徵看到未來資訊
# ----------------------------------------------------------------------
def add_label(df):
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


FEATURE_COLS = [
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    "volatility_10d", "volatility_20d", "ma_gap",
    "volume_change_5d", "volume_ma_ratio", "rsi_14", "day_range",
]


# ----------------------------------------------------------------------
# 4. Walk-forward 驗證
#    用「擴張視窗」訓練,永遠只用過去資料預測未來,避免look-ahead bias
# ----------------------------------------------------------------------
def walk_forward_validate(df, n_splits=6, test_size=60):
    df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index()
    x, y = df[FEATURE_COLS], df["target"]
    n = len(df)

    results: list[pd.DataFrame] = []
    model = None
    fold_start = n - n_splits * test_size

    for i in range(n_splits):
        train_end = fold_start + i * test_size
        test_start = train_end
        test_end = test_start + test_size
        if test_end > n:
            break

        x_train, y_train = x.iloc[:train_end], y.iloc[:train_end]
        x_test, y_test = x.iloc[test_start:test_end], y.iloc[test_start:test_end]

        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        prob = model.predict_proba(x_test)[:, 1]

        fold_result = df.iloc[test_start:test_end][["Date", "Close", "target"]].copy()
        fold_result["pred"] = pred
        fold_result["prob"] = prob
        fold_result["fold"] = i
        results.append(fold_result)

        print(f"Fold {i}: train={train_end}筆, test={test_start}:{test_end}, "
              f"accuracy={accuracy_score(y_test, pred):.3f}, "
              f"precision={precision_score(y_test, pred, zero_division=0):.3f}")

    if model is None:
        raise ValueError(
            f"資料量不足以切出任何一個fold(共{n}筆),"
            f"請減少 n_splits({n_splits}) 或 test_size({test_size})"
        )

    return pd.concat(results).reset_index(drop=True), model


# ----------------------------------------------------------------------
# 5. 簡易回測:把訊號轉成模擬交易,扣除交易成本
# ----------------------------------------------------------------------
def backtest(results_df, cost_bps=5):
    """
    cost_bps: 單邊交易成本(basis points),5bps約略反映台股手續費+稅
    策略邏輯:模型預測漲(1)則持有一天,預測跌(0)則空手
    """
    df = results_df.copy()
    df["actual_return"] = df["Close"].pct_change().shift(-1)  # 隔日實際報酬
    df["strategy_return"] = np.where(df["pred"] == 1, df["actual_return"], 0)

    # 每次訊號改變時計入交易成本
    df["position_change"] = df["pred"].diff().abs().fillna(0)
    df["cost"] = df["position_change"] * (cost_bps / 10000)
    df["strategy_return_net"] = df["strategy_return"] - df["cost"]

    # 最後一筆沒有「隔日」報酬可算,填0避免NaN污染整條累積曲線
    df["strategy_equity"] = (1 + df["strategy_return_net"].fillna(0)).cumprod()
    df["buy_hold_equity"] = (1 + df["actual_return"].fillna(0)).cumprod()

    total_days = len(df)
    ann_factor = 252

    strat_ret = df["strategy_return_net"].dropna()
    sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(ann_factor) if strat_ret.std() > 0 else np.nan
    cum_return = df["strategy_equity"].iloc[-1] - 1
    ann_return = (1 + cum_return) ** (ann_factor / total_days) - 1

    running_max = df["strategy_equity"].cummax()
    drawdown = (df["strategy_equity"] - running_max) / running_max
    max_dd = drawdown.min()

    bh_cum_return = df["buy_hold_equity"].iloc[-1] - 1

    print("\n===== 回測結果(已扣除交易成本)=====")
    print(f"策略累積報酬:     {cum_return:.2%}")
    print(f"Buy & Hold累積報酬: {bh_cum_return:.2%}")
    print(f"策略年化報酬:     {ann_return:.2%}")
    print(f"策略 Sharpe ratio: {sharpe:.2f}")
    print(f"策略最大回撤:     {max_dd:.2%}")

    return df


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    print("抓取資料中...")
    raw = fetch_data()
    print(f"取得 {len(raw)} 筆日資料,期間 {raw.index[0].date()} ~ {raw.index[-1].date()}")

    df = add_features(raw)
    df = add_label(df)

    print("\n開始 walk-forward 驗證訓練...")
    results, last_model = walk_forward_validate(df)

    overall_accuracy = accuracy_score(results["target"], results["pred"])
    overall_precision = precision_score(results["target"], results["pred"], zero_division=0)
    overall_recall = recall_score(results["target"], results["pred"], zero_division=0)
    print(f"\n整體 out-of-sample 準確率: {overall_accuracy:.3f}")
    print(f"整體 precision: {overall_precision:.3f} / recall: {overall_recall:.3f}")
    print("混淆矩陣:\n", confusion_matrix(results["target"], results["pred"]))

    bt = backtest(results)

    # 特徵重要性,幫助你在面試講出「哪些特徵最有貢獻」
    importance = pd.Series(last_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n特徵重要性(最後一個fold的模型):")
    # noinspection PyTypeChecker
    print(importance)

    # 繪製策略 vs buy-and-hold 累積報酬
    # noinspection PyTypeChecker
    plt.figure(figsize=(10, 5))
    plt.plot(bt["Date"], bt["strategy_equity"], label="Model Strategy")
    plt.plot(bt["Date"], bt["buy_hold_equity"], label="Buy & Hold")
    plt.legend()
    plt.title(f"{TICKER} Strategy vs Buy & Hold (Out-of-sample)")
    plt.ylabel("Cumulative Return (x)")
    plt.tight_layout()
    plt.savefig("outputs/equity_curve.png", dpi=150)
    print("\n已儲存 equity_curve.png")


if __name__ == "__main__":
    main()
