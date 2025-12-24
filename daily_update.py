import os
import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import ccxt

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from arch import arch_model
import ruptures as rpt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(DATA_DIR, "daily_results.csv")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")

FALLBACK_EXCHANGES = ["kraken", "coinbase", "bitstamp"]
GRAPH_NETWORK_SUBGRAPH_ID = "GgwLf9BTFBJi6Z5iYHssMAGEE4w5dR3Jox2dMLrBxnCT"

DEFAULT_SETTINGS = {
    "preferred_exchange": "binance",
    "symbol": "GRT/USDT",
    "benchmark": "BTC/USDT",
    "days": 900,
    "timeframe": "1d",
    "api_keys": {"thegraph_gateway": ""}
}


def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_settings():
    if not os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2, ensure_ascii=False)
        return DEFAULT_SETTINGS
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _symbol_variants(symbol: str):
    base, quote = symbol.split("/")
    variants = [symbol]
    if quote.upper() == "USDT":
        variants += [f"{base}/USD", f"{base}/USDC"]
    return variants


def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int):
    exchanges_to_try = [exchange_id] + [ex for ex in FALLBACK_EXCHANGES if ex != exchange_id]
    last_err = None

    for ex_id in exchanges_to_try:
        try:
            ex_class = getattr(ccxt, ex_id)
            ex = ex_class({"enableRateLimit": True})
            ex.load_markets()

            for sym in _symbol_variants(symbol):
                if sym not in ex.markets:
                    continue

                ohlcv = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
                df = df.drop(columns=["timestamp"]).set_index("date")
                df.attrs["exchange_used"] = ex_id
                df.attrs["symbol_used"] = sym
                return df

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if ("451" in msg) or ("restricted location" in msg) or ("eligibility" in msg):
                continue
            continue

    raise RuntimeError(f"No se pudo descargar OHLCV. Último error: {last_err}")


def compute_returns(df):
    df = df.copy()
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["log_close"] = np.log(df["close"])
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = df["log_close"].diff()
    return df


def safe_float(x):
    try:
        if x is None:
            return np.nan
        v = float(x)
        if np.isinf(v):
            return np.nan
        return v
    except Exception:
        return np.nan


def _graphql_post(url: str, query: str, variables=None, timeout=25) -> dict:
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_grt_network_fundamentals(thegraph_key: str) -> pd.DataFrame:
    if not thegraph_key:
        return pd.DataFrame()

    url = f"https://gateway.thegraph.com/api/{thegraph_key}/subgraphs/id/{GRAPH_NETWORK_SUBGRAPH_ID}"
    q = """
    query {
      graphNetwork(id: "1") {
        id
        totalTokensStaked
        totalTokensAllocated
        totalDelegatedTokens
        totalSupply
      }
      indexers(first: 1000, where: {active: true}) { id }
    }
    """
    data = _graphql_post(url, q)
    if "errors" in data:
        return pd.DataFrame()

    d = data["data"]
    gn = d.get("graphNetwork") or {}
    idx_count = len(d.get("indexers") or [])

    row = {
        "as_of": str(pd.Timestamp.utcnow().date()),
        "totalTokensStaked": safe_float(gn.get("totalTokensStaked")),
        "totalSupply": safe_float(gn.get("totalSupply")),
        "activeIndexers": float(idx_count),
    }
    return pd.DataFrame([row]).set_index("as_of")


def trend_regression(df, window=90):
    d = df.dropna().tail(window).copy()
    y = d["log_close"].values
    x = np.arange(len(d))
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return {"trend_slope": float(model.params[1]), "trend_pvalue": float(model.pvalues[1])}


def momentum_metrics(df):
    r = df["ret"].dropna()
    r30 = float((1 + r.tail(30)).prod() - 1) if len(r) >= 30 else np.nan
    return {"mom_ret_30d": r30}


def volume_signal(df):
    v = df["volume"]
    hist = v.tail(180)
    recent = v.tail(14).mean()
    z = float((recent - hist.mean()) / (hist.std() + 1e-12)) if len(hist) > 30 else np.nan
    return {"vol_z_14d": z}


def garch_volatility(df):
    r = df["ret"].dropna() * 100.0
    if len(r) < 250:
        return {"garch_vol_now": np.nan}
    am = arch_model(r, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
    res = am.fit(disp="off")
    return {"garch_vol_now": float(res.conditional_volatility.iloc[-1])}


def build_feature_frame(df, bench, fund_last: Optional[pd.Series]):
    d = df.copy()
    d["ret_7"] = d["close"].pct_change(7)
    d["ret_30"] = d["close"].pct_change(30)
    d["vol_14"] = d["ret"].rolling(14).std()
    d["mom_30"] = (1 + d["ret"]).rolling(30).apply(lambda x: np.prod(1 + x) - 1, raw=False)

    tmp = pd.concat([d["ret"].rename("asset"), bench["ret"].rename("bench")], axis=1)
    d["corr_60"] = tmp["asset"].rolling(60).corr(tmp["bench"])

    if fund_last is not None and not fund_last.empty:
        d["fund_totalTokensStaked"] = safe_float(fund_last.get("totalTokensStaked"))
        d["fund_totalSupply"] = safe_float(fund_last.get("totalSupply"))
        d["fund_stake_ratio"] = d["fund_totalTokensStaked"] / (d["fund_totalSupply"] + 1e-12)

    return d


def fit_probs(feat, horizons=(7,30,90), min_rows=250):
    drop_cols = {"open","high","low","close","volume","log_close","ret","log_ret"}
    candidates = [c for c in feat.columns if c not in drop_cols]
    X_all = feat[candidates].apply(pd.to_numeric, errors="coerce")
    X_all = X_all.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    probs = {}
    aucs = {}

    for h in horizons:
        y = ((feat["close"].shift(-h) / feat["close"]) - 1.0) > 0
        y = y.astype(float)

        data = pd.concat([X_all, y.rename("y")], axis=1).dropna()
        if len(data) < min_rows:
            probs[h] = np.nan
            aucs[h] = np.nan
            continue

        n = len(data)
        cut = int(n * 0.8)
        train = data.iloc[:cut]
        test = data.iloc[cut:]

        X_train = train.drop(columns=["y"])
        y_train = train["y"].astype(int)
        X_test = test.drop(columns=["y"])
        y_test = test["y"].astype(int)

        logit = Pipeline([("scaler", StandardScaler(with_mean=False)),
                          ("clf", LogisticRegression(max_iter=500, n_jobs=1))])
        gbt = GradientBoostingClassifier(random_state=7)

        logit.fit(X_train, y_train)
        gbt.fit(X_train, y_train)

        p_test = 0.5*logit.predict_proba(X_test)[:,1] + 0.5*gbt.predict_proba(X_test)[:,1]
        aucs[h] = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else np.nan

        last_row = X_all.iloc[[-1]]
        probs[h] = float(0.5*logit.predict_proba(last_row)[:,1][0] + 0.5*gbt.predict_proba(last_row)[:,1][0])

    return probs, aucs


def grt_score(metrics, probs):
    s = 50.0
    slope = metrics.get("trend_slope", np.nan)
    pval = metrics.get("trend_pvalue", np.nan)
    if slope == slope and pval == pval:
        s += 10 if (slope > 0 and pval < 0.05) else (-10 if (slope < 0 and pval < 0.05) else 0)

    m30 = metrics.get("mom_ret_30d", np.nan)
    if m30 == m30:
        s += 8 if m30 > 0 else -8

    p30 = probs.get(30, np.nan)
    if p30 == p30:
        s += 8 if p30 > 0.6 else (-8 if p30 < 0.45 else 0)

    return float(np.clip(s, 0, 100))


def save_row(row: dict):
    df = pd.DataFrame([row])
    if os.path.exists(RESULTS_PATH):
        old = pd.read_csv(RESULTS_PATH)
        out = pd.concat([old, df], ignore_index=True)
        out = out.drop_duplicates(subset=["as_of_date","exchange","symbol"], keep="last")
    else:
        out = df
    out.to_csv(RESULTS_PATH, index=False)


def main():
    s = load_settings()
    exchange = s.get("preferred_exchange","binance")
    symbol = s.get("symbol","GRT/USDT")
    benchmark = s.get("benchmark","BTC/USDT")
    days = int(s.get("days", 900))
    tf = s.get("timeframe","1d")
    tgk = s.get("api_keys", {}).get("thegraph_gateway", "")

    df = compute_returns(fetch_ohlcv(exchange, symbol, tf, limit=days))
    dfb = compute_returns(fetch_ohlcv(exchange, benchmark, tf, limit=days))

    fund = fetch_grt_network_fundamentals(tgk)
    fund_last = fund.iloc[-1] if not fund.empty else None

    metrics = {}
    metrics.update(trend_regression(df, 90))
    metrics.update(momentum_metrics(df))
    metrics.update(volume_signal(df))
    metrics.update(garch_volatility(df))

    feat = build_feature_frame(df, dfb, fund_last)
    probs, aucs = fit_probs(feat, horizons=(7,30,90), min_rows=250)
    score = grt_score(metrics, probs)

    row = {
        "as_of_date": str(df.index.max()),
        "exchange": df.attrs.get("exchange_used", exchange),
        "symbol": df.attrs.get("symbol_used", symbol),
        "benchmark": dfb.attrs.get("symbol_used", benchmark),
        "updated_at_utc": now_utc_iso(),
        "score_0_100": float(score),
        "p_up_7d": probs.get(7, np.nan),
        "p_up_30d": probs.get(30, np.nan),
        "p_up_90d": probs.get(90, np.nan),
        "auc_7d": aucs.get(7, np.nan),
        "auc_30d": aucs.get(30, np.nan),
        "auc_90d": aucs.get(90, np.nan),
        **metrics
    }
    save_row(row)


if __name__ == "__main__":
    main()
