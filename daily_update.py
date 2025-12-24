# daily_update.py
import os
import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import ccxt

from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from arch import arch_model
import ruptures as rpt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# HMM opcional
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
GRAPH_NETWORK_SUBGRAPH = "https://api.thegraph.com/subgraphs/name/graphprotocol/graph-network-mainnet"


DEFAULT_SETTINGS = {
    "preferred_exchange": "binance",
    "symbol": "GRT/USDT",
    "benchmark": "BTC/USDT",
    "days": 900,
    "timeframe": "1d",
    "api_keys": {}
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
    df["log_close"] = np.log(df["close"].astype(float))
    df["ret"] = df["close"].astype(float).pct_change()
    df["log_ret"] = df["log_close"].diff()
    return df


def _graphql_post(url: str, query: str, variables=None, timeout=20) -> dict:
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_grt_network_fundamentals() -> pd.DataFrame:
    q = """
    query {
      graphNetwork(id: "1") {
        id
        totalTokensStaked
        totalTokensAllocated
        totalDelegatedTokens
        totalSupply
        totalQueryFees
        totalIndexerRewards
        totalDelegationRewards
      }
      indexers(first: 1000, where: {active: true}) { id }
      delegators(first: 1000) { id }
    }
    """
    data = _graphql_post(GRAPH_NETWORK_SUBGRAPH, q)
    if "errors" in data:
        raise RuntimeError(f"Graph subgraph error: {data['errors']}")
    d = data["data"]

    gn = d.get("graphNetwork") or {}
    idx_count = len(d.get("indexers") or [])
    del_count = len(d.get("delegators") or [])

    row = {
        "as_of": str(pd.Timestamp.utcnow().date()),
        "totalTokensStaked": float(gn.get("totalTokensStaked")) if gn.get("totalTokensStaked") else np.nan,
        "totalTokensAllocated": float(gn.get("totalTokensAllocated")) if gn.get("totalTokensAllocated") else np.nan,
        "totalDelegatedTokens": float(gn.get("totalDelegatedTokens")) if gn.get("totalDelegatedTokens") else np.nan,
        "totalSupply": float(gn.get("totalSupply")) if gn.get("totalSupply") else np.nan,
        "totalQueryFees": float(gn.get("totalQueryFees")) if gn.get("totalQueryFees") else np.nan,
        "totalIndexerRewards": float(gn.get("totalIndexerRewards")) if gn.get("totalIndexerRewards") else np.nan,
        "totalDelegationRewards": float(gn.get("totalDelegationRewards")) if gn.get("totalDelegationRewards") else np.nan,
        "activeIndexers": float(idx_count),
        "delegatorsApprox": float(del_count),
    }
    return pd.DataFrame([row]).set_index("as_of")


def trend_regression(df, window=90):
    d = df.dropna().tail(window).copy()
    y = d["log_close"].values
    x = np.arange(len(d))
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return {"trend_slope": float(model.params[1]), "trend_pvalue": float(model.pvalues[1]), "trend_r2": float(model.rsquared)}


def stationarity_tests(df):
    d = df["log_ret"].dropna()
    adf = adfuller(d, autolag="AIC")
    kpss_stat, kpss_p, _, _ = kpss(d, regression="c", nlags="auto")
    return {"adf_stat": float(adf[0]), "adf_pvalue": float(adf[1]), "kpss_stat": float(kpss_stat), "kpss_pvalue": float(kpss_p)}


def momentum_metrics(df):
    r = df["ret"].dropna()
    r30 = float((1 + r.tail(30)).prod() - 1) if len(r) >= 30 else np.nan
    r90 = float((1 + r.tail(90)).prod() - 1) if len(r) >= 90 else np.nan
    recent = r.tail(14)
    z = float((recent.mean() - r.mean()) / (r.std() + 1e-12)) if len(r) > 60 else np.nan
    return {"mom_ret_30d": r30, "mom_ret_90d": r90, "mom_z_14d": z}


def volume_signal(df):
    v = df["volume"].astype(float)
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


def tail_risk_metrics(df):
    r = df["ret"].dropna()
    if len(r) < 250:
        return {"skew_90d": np.nan, "kurt_90d": np.nan, "downside_vol_90d": np.nan}
    w = r.tail(90)
    skew = float(w.skew())
    kurt = float(w.kurt())
    downside = w[w < 0]
    downside_vol = float(downside.std()) if len(downside) > 10 else np.nan
    return {"skew_90d": skew, "kurt_90d": kurt, "downside_vol_90d": downside_vol}


def structural_breaks(df):
    y = df["log_close"].dropna().values
    if len(y) < 180:
        return {"breakpoints_n": np.nan, "break_recent": np.nan}
    algo = rpt.Pelt(model="rbf").fit(y)
    bkps = algo.predict(pen=8)
    b = bkps[:-1]
    n = len(b)
    recent = any((len(y) - bp) <= 30 for bp in b) if n > 0 else False
    return {"breakpoints_n": float(n), "break_recent": float(1.0 if recent else 0.0)}


def rolling_correlation(df_asset, df_bench, window=60):
    joined = pd.concat([df_asset["ret"].rename("asset"), df_bench["ret"].rename("bench")], axis=1).dropna()
    if len(joined) < window + 10:
        return {"corr_60d": np.nan}
    return {"corr_60d": float(joined["asset"].rolling(window).corr(joined["bench"]).iloc[-1])}


def hmm_regimes(df):
    if not HMM_AVAILABLE:
        return {"hmm_regime": None, "hmm_p_regime": np.nan}

    r = df["ret"].dropna()
    if len(r) < 400:
        return {"hmm_regime": None, "hmm_p_regime": np.nan}

    vol = r.rolling(14).std()
    x = pd.concat([r, vol], axis=1).dropna()
    X = x.values

    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=250, random_state=7)
    model.fit(X)

    post = model.predict_proba(X)
    current_state = int(np.argmax(post[-1]))
    p_state = float(np.max(post[-1]))

    states = model.predict(X)
    means = []
    for s in range(3):
        means.append(float(np.mean(r.loc[x.index][states == s])))
    order = np.argsort(means)
    label_map = {int(order[0]): "Dump / Risk", int(order[1]): "Range / Accum", int(order[2]): "Uptrend"}
    label = label_map.get(current_state, "Unknown")
    return {"hmm_regime": label, "hmm_p_regime": p_state}


def build_feature_frame(df, bench, fund: Optional[pd.DataFrame]):
    d = df.copy()
    d["ret_1"] = d["ret"]
    d["ret_7"] = d["close"].pct_change(7)
    d["ret_30"] = d["close"].pct_change(30)
    d["vol_14"] = d["ret"].rolling(14).std()
    d["vol_60"] = d["ret"].rolling(60).std()
    d["mom_30"] = (1 + d["ret"]).rolling(30).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    d["vol_z_14"] = (d["volume"].rolling(14).mean() - d["volume"].rolling(180).mean()) / (d["volume"].rolling(180).std() + 1e-12)

    j = pd.concat([d["ret"], bench["ret"].rename("bench_ret")], axis=1).dropna()
    d["corr_60"] = j["ret"].rolling(60).corr(j["bench_ret"])

    d["skew_90"] = d["ret"].rolling(90).skew()
    d["kurt_90"] = d["ret"].rolling(90).kurt()

    if fund is not None and not fund.empty:
        f = fund.iloc[-1].to_dict()
        for k, v in f.items():
            d[f"fund_{k}"] = v
        if "fund_totalTokensStaked" in d.columns and "fund_totalSupply" in d.columns:
            d["fund_stake_ratio"] = d["fund_totalTokensStaked"] / (d["fund_totalSupply"] + 1e-12)

    return d


def fit_probs(feat, horizons=(7,30,90)):
    feat = feat.copy()
    candidates = [c for c in feat.columns if c not in ("open","high","low","close","volume","log_close","ret","log_ret")]
    X_all = feat[candidates].apply(pd.to_numeric, errors="coerce")

    probs = {}
    aucs = {}

    for h in horizons:
        y = (feat["close"].shift(-h) / feat["close"] - 1.0) > 0
        y = y.astype(float)
        data = pd.concat([X_all, y.rename("y")], axis=1).dropna()
        if len(data) < 600:
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

        logit = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=500, n_jobs=1))
        ])
        gbt = GradientBoostingClassifier(random_state=7)

        logit.fit(X_train, y_train)
        gbt.fit(X_train, y_train)

        p_test = 0.5 * logit.predict_proba(X_test)[:,1] + 0.5 * gbt.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else np.nan

        last_row = X_all.dropna().iloc[[-1]]
        p_today = 0.5 * logit.predict_proba(last_row)[:,1][0] + 0.5 * gbt.predict_proba(last_row)[:,1][0]

        probs[h] = float(p_today)
        aucs[h] = float(auc) if auc == auc else np.nan

    return probs, aucs


def grt_score(metrics: dict, probs: dict) -> float:
    s = 50.0
    slope = metrics.get("trend_slope", np.nan)
    pval = metrics.get("trend_pvalue", np.nan)
    if slope == slope and pval == pval:
        s += 14 if (slope > 0 and pval < 0.05) else (7 if slope > 0 else (-14 if (slope < 0 and pval < 0.05) else (-7 if slope < 0 else 0)))

    m30 = metrics.get("mom_ret_30d", np.nan)
    if m30 == m30:
        s += 10 if m30 > 0 else -10

    vz = metrics.get("vol_z_14d", np.nan)
    if vz == vz:
        s += 7 if vz > 0.5 else (-7 if vz < -0.5 else 0)

    gv = metrics.get("garch_vol_now", np.nan)
    if gv == gv:
        s += 3 if gv < 4 else (-6 if gv > 9 else -2)

    skew = metrics.get("skew_90d", np.nan)
    if skew == skew:
        s += 4 if skew > 0.2 else (-4 if skew < -0.2 else 0)

    corr = metrics.get("corr_60d", np.nan)
    if corr == corr:
        s += 3 if corr < 0.5 else 0

    regime = metrics.get("hmm_regime", None)
    p_reg = metrics.get("hmm_p_regime", np.nan)
    if regime and isinstance(regime, str):
        if "Uptrend" in regime:
            s += 8 if (p_reg == p_reg and p_reg > 0.6) else 5
        elif "Range" in regime:
            s += 2
        elif "Dump" in regime:
            s -= 8

    stake_ratio = metrics.get("fund_stake_ratio", np.nan)
    if stake_ratio == stake_ratio:
        s += 6 if stake_ratio > 0.35 else (3 if stake_ratio > 0.25 else 0)

    p30 = probs.get(30, np.nan)
    p90 = probs.get(90, np.nan)
    if p30 == p30:
        s += 8 if p30 > 0.6 else (-8 if p30 < 0.45 else 0)
    if p90 == p90:
        s += 6 if p90 > 0.6 else (-6 if p90 < 0.45 else 0)

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

    df = compute_returns(fetch_ohlcv(exchange, symbol, tf, limit=days))
    dfb = compute_returns(fetch_ohlcv(exchange, benchmark, tf, limit=days))

    # Fundamentals
    try:
        fund = fetch_grt_network_fundamentals()
    except Exception:
        fund = pd.DataFrame()

    metrics = {}
    metrics.update(trend_regression(df, 90))
    metrics.update(stationarity_tests(df))
    metrics.update(momentum_metrics(df))
    metrics.update(volume_signal(df))
    metrics.update(garch_volatility(df))
    metrics.update(structural_breaks(df))
    metrics.update(rolling_correlation(df, dfb, 60))
    metrics.update(tail_risk_metrics(df))
    metrics.update(hmm_regimes(df))

    if not fund.empty:
        f = fund.iloc[-1].to_dict()
        for k, v in f.items():
            metrics[f"fund_{k}"] = v
        ts = metrics.get("fund_totalTokensStaked", np.nan)
        sup = metrics.get("fund_totalSupply", np.nan)
        if ts == ts and sup == sup:
            metrics["fund_stake_ratio"] = float(ts / (sup + 1e-12))

    feat = build_feature_frame(df, dfb, fund if not fund.empty else None)
    probs, aucs = fit_probs(feat, horizons=(7,30,90))
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
