import os
import json
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

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
from sklearn.inspection import permutation_importance

# HMM opcional
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

import plotly.graph_objects as go
import plotly.express as px


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="GRT QuantLab", page_icon="🟣", layout="wide")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(DATA_DIR, "daily_results.csv")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")

FALLBACK_EXCHANGES = ["kraken", "coinbase", "bitstamp"]

# Subgraph ID de "Graph Network Mainnet" (desde Explorer)
# Fuente: Graph Explorer muestra Query URL /subgraphs/id/<ID> :contentReference[oaicite:2]{index=2}
GRAPH_NETWORK_SUBGRAPH_ID = "GgwLf9BTFBJi6Z5iYHssMAGEE4w5dR3Jox2dMLrBxnCT"

DEFAULT_SETTINGS = {
    "preferred_exchange": "binance",
    "symbol": "GRT/USDT",
    "benchmark": "BTC/USDT",
    "days": 900,
    "timeframe": "1d",
    "api_keys": {
        "thegraph_gateway": ""   # <-- pon aquí tu API key del Graph Gateway
    }
}


# =========================
# UI THEME (épico)
# =========================
EPIC_CSS = """
<style>
:root{
  --bg0:#070A12;
  --bg1:#0B1020;
  --card:#0F1730;
  --text:#E8EEFF;
  --muted:#AAB6E8;
  --accent:#A855F7;
  --line:#1E2A55;
}
html, body, [class*="css"]  {
  background: radial-gradient(1200px 600px at 20% -10%, rgba(168,85,247,0.35), transparent 60%),
              radial-gradient(900px 500px at 110% 10%, rgba(34,197,94,0.20), transparent 55%),
              linear-gradient(180deg, var(--bg0), var(--bg1));
  color: var(--text) !important;
}
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(15,23,48,0.95), rgba(11,16,32,0.95)) !important;
  border-right: 1px solid rgba(30,42,85,0.7);
}
.block-container{ padding-top: 1.1rem; }
.epic-hero{
  padding: 18px 18px; border-radius: 18px;
  background: linear-gradient(135deg, rgba(168,85,247,0.20), rgba(34,197,94,0.08));
  border: 1px solid rgba(30,42,85,0.7);
  box-shadow: 0 18px 60px rgba(0,0,0,0.35);
}
.epic-title{ font-size: 26px; font-weight: 800; letter-spacing: -0.3px; }
.epic-sub{ color: var(--muted); font-size: 14px; }
.kpi{
  border-radius: 16px; padding: 14px 14px;
  background: linear-gradient(180deg, rgba(15,23,48,0.95), rgba(16,27,59,0.85));
  border: 1px solid rgba(30,42,85,0.7);
}
.kpi .label{ color: var(--muted); font-size: 12px; margin-bottom: 6px; }
.kpi .value{ font-size: 20px; font-weight: 800; }
.kpi .hint{ color: var(--muted); font-size: 11px; margin-top: 6px; }
.badge{
  display:inline-block; padding: 4px 10px; border-radius: 999px;
  background: rgba(168,85,247,0.15);
  border:1px solid rgba(168,85,247,0.35);
  color: var(--text); font-size: 12px; margin-left: 8px;
}
</style>
"""
st.markdown(EPIC_CSS, unsafe_allow_html=True)


# =========================
# HELPERS
# =========================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_settings() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2, ensure_ascii=False)
        return DEFAULT_SETTINGS
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_settings(s: dict) -> None:
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)


def _symbol_variants(symbol: str):
    base, quote = symbol.split("/")
    variants = [symbol]
    if quote.upper() == "USDT":
        variants += [f"{base}/USD", f"{base}/USDC"]
    return variants


@st.cache_data(ttl=60 * 60)
def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
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


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
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


@st.cache_data(ttl=6 * 60 * 60)
def fetch_grt_network_fundamentals(thegraph_api_key: str) -> pd.DataFrame:
    """
    Usa Graph Gateway:
    https://gateway.thegraph.com/api/<KEY>/subgraphs/id/<SUBGRAPH_ID> :contentReference[oaicite:3]{index=3}
    """
    if not thegraph_api_key:
        raise RuntimeError("Falta THE_GRAPH_API_KEY (Gateway).")

    url = f"https://gateway.thegraph.com/api/{thegraph_api_key}/subgraphs/id/{GRAPH_NETWORK_SUBGRAPH_ID}"

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

    data = _graphql_post(url, q)
    if "errors" in data:
        raise RuntimeError(f"Graph gateway error: {data['errors']}")

    d = data["data"]
    gn = d.get("graphNetwork") or {}
    idx_count = len(d.get("indexers") or [])
    del_count = len(d.get("delegators") or [])

    row = {
        "as_of": str(pd.Timestamp.utcnow().date()),
        "totalTokensStaked": safe_float(gn.get("totalTokensStaked")),
        "totalTokensAllocated": safe_float(gn.get("totalTokensAllocated")),
        "totalDelegatedTokens": safe_float(gn.get("totalDelegatedTokens")),
        "totalSupply": safe_float(gn.get("totalSupply")),
        "totalQueryFees": safe_float(gn.get("totalQueryFees")),
        "totalIndexerRewards": safe_float(gn.get("totalIndexerRewards")),
        "totalDelegationRewards": safe_float(gn.get("totalDelegationRewards")),
        "activeIndexers": float(idx_count),
        "delegatorsApprox": float(del_count),
    }
    return pd.DataFrame([row]).set_index("as_of")


# =========================
# METRICS
# =========================
def trend_regression(df: pd.DataFrame, window: int = 90) -> dict:
    d = df.dropna().tail(window).copy()
    y = d["log_close"].values
    x = np.arange(len(d))
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return {
        "trend_slope": float(model.params[1]),
        "trend_pvalue": float(model.pvalues[1]),
        "trend_r2": float(model.rsquared),
    }


def stationarity_tests(df: pd.DataFrame) -> dict:
    d = df["log_ret"].dropna()
    adf = adfuller(d, autolag="AIC")
    kpss_stat, kpss_p, _, _ = kpss(d, regression="c", nlags="auto")
    return {
        "adf_pvalue": float(adf[1]),
        "kpss_pvalue": float(kpss_p),
    }


def momentum_metrics(df: pd.DataFrame) -> dict:
    r = df["ret"].dropna()
    r30 = float((1 + r.tail(30)).prod() - 1) if len(r) >= 30 else np.nan
    r90 = float((1 + r.tail(90)).prod() - 1) if len(r) >= 90 else np.nan
    return {"mom_ret_30d": r30, "mom_ret_90d": r90}


def volume_signal(df: pd.DataFrame) -> dict:
    v = df["volume"]
    hist = v.tail(180)
    recent = v.tail(14).mean()
    z = float((recent - hist.mean()) / (hist.std() + 1e-12)) if len(hist) > 30 else np.nan
    return {"vol_z_14d": z}


def garch_volatility(df: pd.DataFrame) -> dict:
    r = df["ret"].dropna() * 100.0
    if len(r) < 250:
        return {"garch_vol_now": np.nan}
    am = arch_model(r, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
    res = am.fit(disp="off")
    return {"garch_vol_now": float(res.conditional_volatility.iloc[-1])}


def tail_risk_metrics(df: pd.DataFrame) -> dict:
    r = df["ret"].dropna()
    if len(r) < 250:
        return {"skew_90d": np.nan, "kurt_90d": np.nan, "downside_vol_90d": np.nan}
    w = r.tail(90)
    downside = w[w < 0]
    return {
        "skew_90d": float(w.skew()),
        "kurt_90d": float(w.kurt()),
        "downside_vol_90d": float(downside.std()) if len(downside) > 10 else np.nan,
    }


def structural_breaks(df: pd.DataFrame) -> dict:
    y = df["log_close"].dropna().values
    if len(y) < 180:
        return {"breakpoints_n": np.nan}
    algo = rpt.Pelt(model="rbf").fit(y)
    bkps = algo.predict(pen=8)
    return {"breakpoints_n": float(max(0, len(bkps) - 1))}


def hmm_regimes(df: pd.DataFrame) -> dict:
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
    return {"hmm_regime": label_map.get(current_state, "Unknown"), "hmm_p_regime": p_state}


# =========================
# FEATURES + MODELOS (FIX)
# =========================
def build_feature_frame(df: pd.DataFrame, bench: pd.DataFrame, fund_last: Optional[pd.Series]) -> pd.DataFrame:
    d = df.copy()

    d["ret_1"] = d["ret"]
    d["ret_7"] = d["close"].pct_change(7)
    d["ret_30"] = d["close"].pct_change(30)
    d["vol_14"] = d["ret"].rolling(14).std()
    d["vol_60"] = d["ret"].rolling(60).std()
    d["mom_30"] = (1 + d["ret"]).rolling(30).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    d["vol_z_14"] = (d["volume"].rolling(14).mean() - d["volume"].rolling(180).mean()) / (d["volume"].rolling(180).std() + 1e-12)

    # Correlación robusta (merge por índice)
    tmp = pd.concat([d["ret"].rename("asset"), bench["ret"].rename("bench")], axis=1)
    d["corr_60"] = tmp["asset"].rolling(60).corr(tmp["bench"])

    d["skew_90"] = d["ret"].rolling(90).skew()
    d["kurt_90"] = d["ret"].rolling(90).kurt()

    # Fundamentals (as-of)
    if fund_last is not None and not fund_last.empty:
        for k, v in fund_last.to_dict().items():
            d[f"fund_{k}"] = safe_float(v)
        ts = d.get("fund_totalTokensStaked")
        sup = d.get("fund_totalSupply")
        if ts is not None and sup is not None:
            d["fund_stake_ratio"] = d["fund_totalTokensStaked"] / (d["fund_totalSupply"] + 1e-12)

    return d


def fit_ensemble_probabilities(feat: pd.DataFrame, horizons=(7, 30, 90), min_rows=250) -> Tuple[dict, dict, dict]:
    """
    FIXES:
    - imputa NaNs (ffill/bfill) para evitar dataset vacío
    - baja el mínimo de filas a min_rows
    - devuelve diagnóstico de tamaños reales
    """
    feat = feat.copy()

    drop_cols = {"open","high","low","close","volume","log_close","ret","log_ret"}
    candidates = [c for c in feat.columns if c not in drop_cols]

    X_all = feat[candidates].apply(pd.to_numeric, errors="coerce")
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    X_all = X_all.ffill().bfill()  # <- clave para que no se quede en 0

    probs = {}
    report = {}
    diag = {}

    for h in horizons:
        y = ((feat["close"].shift(-h) / feat["close"]) - 1.0) > 0
        y = y.astype(float)

        data = pd.concat([X_all, y.rename("y")], axis=1).dropna()
        diag[h] = {"rows_after_clean": int(len(data))}

        if len(data) < min_rows:
            probs[h] = np.nan
            report[h] = {"auc": np.nan, "n_train": 0, "n_test": 0, "top_features": []}
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

        p_logit = logit.predict_proba(X_test)[:, 1]
        p_gbt = gbt.predict_proba(X_test)[:, 1]
        p_ens = 0.5 * p_logit + 0.5 * p_gbt

        auc = roc_auc_score(y_test, p_ens) if len(np.unique(y_test)) > 1 else np.nan

        last_row = X_all.iloc[[-1]]
        p_today = 0.5 * logit.predict_proba(last_row)[:, 1][0] + 0.5 * gbt.predict_proba(last_row)[:, 1][0]
        probs[h] = float(p_today)

        # Explainability
        try:
            imp = permutation_importance(gbt, X_test, y_test, n_repeats=7, random_state=7, scoring="roc_auc")
            imp_df = pd.DataFrame({"feature": X_test.columns, "importance": imp.importances_mean})
            top = imp_df.sort_values("importance", ascending=False).head(10)
            top_feats = list(zip(top["feature"].tolist(), top["importance"].astype(float).round(4).tolist()))
        except Exception:
            top_feats = []

        report[h] = {"auc": float(auc) if auc == auc else np.nan, "n_train": int(len(train)), "n_test": int(len(test)), "top_features": top_feats}

    return probs, report, diag


def grt_score(metrics: dict, probs: dict) -> float:
    s = 50.0

    slope = metrics.get("trend_slope", np.nan)
    pval = metrics.get("trend_pvalue", np.nan)
    if slope == slope and pval == pval:
        if slope > 0 and pval < 0.05:
            s += 14
        elif slope > 0:
            s += 7
        elif slope < 0 and pval < 0.05:
            s -= 14
        elif slope < 0:
            s -= 7

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

    regime = metrics.get("hmm_regime", None)
    p_reg = metrics.get("hmm_p_regime", np.nan)
    if regime and isinstance(regime, str):
        if "Uptrend" in regime:
            s += 8 if (p_reg == p_reg and p_reg > 0.6) else 5
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


def save_row_to_csv(row: dict, path: str = RESULTS_PATH):
    df = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        out = pd.concat([old, df], ignore_index=True)
        out = out.drop_duplicates(subset=["as_of_date","exchange","symbol"], keep="last")
    else:
        out = df
    out.to_csv(path, index=False)


def load_results(path: str = RESULTS_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


# =========================
# SIDEBAR
# =========================
settings = load_settings()

with st.sidebar:
    st.markdown("## 🟣 GRT QuantLab")
    st.caption("Regímenes · Probabilidades · Fundamentals (Gateway)")

    preferred_exchange = st.selectbox(
        "Exchange preferido (fallback auto)",
        ["binance", "kraken", "coinbase", "bitstamp"],
        index=["binance","kraken","coinbase","bitstamp"].index(settings.get("preferred_exchange","binance"))
    )
    symbol = st.text_input("Símbolo", value=settings.get("symbol","GRT/USDT"))
    benchmark = st.text_input("Benchmark", value=settings.get("benchmark","BTC/USDT"))
    days = st.slider("Histórico (días)", 250, 2000, int(settings.get("days", 900)), step=50)

    st.divider()
    st.markdown("### 🔌 The Graph Gateway")
    st.caption("Necesitas API key para que carguen los fundamentals.")
    api_keys = settings.get("api_keys", {})
    thegraph_key = st.text_input("THE_GRAPH_API_KEY", value=api_keys.get("thegraph_gateway",""), type="password")

    auto_save = st.toggle("Guardar resultado al actualizar", value=True)
    run_update = st.button("🔄 Actualizar (calcular todo)", type="primary")

    if st.button("💾 Guardar ajustes"):
        settings["preferred_exchange"] = preferred_exchange
        settings["symbol"] = symbol.strip()
        settings["benchmark"] = benchmark.strip()
        settings["days"] = int(days)
        settings.setdefault("api_keys", {})
        settings["api_keys"]["thegraph_gateway"] = thegraph_key.strip()
        save_settings(settings)
        st.success("Ajustes guardados.")


# =========================
# HERO
# =========================
st.markdown(
    """
    <div class="epic-hero">
      <div class="epic-title">🟣 GRT QuantLab <span class="badge">Regímenes · Probabilidades · Fundamentals</span></div>
      <div class="epic-sub">
        Modelos (7/30/90) + explicabilidad + fundamentals de red via Graph Gateway.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# =========================
# COMPUTE
# =========================
metrics = {}
probs = {7: np.nan, 30: np.nan, 90: np.nan}
report = {}
diag = {}
df = pd.DataFrame()
dfb = pd.DataFrame()
fund = pd.DataFrame()
score = np.nan

if run_update:
    try:
        tf = settings.get("timeframe","1d")
        df = compute_returns(fetch_ohlcv(preferred_exchange, symbol, tf, limit=int(days)))
        dfb = compute_returns(fetch_ohlcv(preferred_exchange, benchmark, tf, limit=int(days)))

        used_ex = df.attrs.get("exchange_used", preferred_exchange)
        used_sym = df.attrs.get("symbol_used", symbol)
        used_bench = dfb.attrs.get("symbol_used", benchmark)

        # Fundamentals (Gateway)
        try:
            fund = fetch_grt_network_fundamentals(settings.get("api_keys", {}).get("thegraph_gateway",""))
        except Exception as fe:
            fund = pd.DataFrame()
            st.warning(f"Fundamentals no disponibles: {fe}")

        # Metrics
        metrics.update(trend_regression(df, 90))
        metrics.update(stationarity_tests(df))
        metrics.update(momentum_metrics(df))
        metrics.update(volume_signal(df))
        metrics.update(garch_volatility(df))
        metrics.update(structural_breaks(df))
        metrics.update(tail_risk_metrics(df))
        metrics.update(hmm_regimes(df))

        fund_last = fund.iloc[-1] if (fund is not None and not fund.empty) else None
        feat = build_feature_frame(df, dfb, fund_last)
        probs, report, diag = fit_ensemble_probabilities(feat, horizons=(7,30,90), min_rows=250)

        # ratio stake
        if fund_last is not None:
            ts = safe_float(fund_last.get("totalTokensStaked"))
            sup = safe_float(fund_last.get("totalSupply"))
            if ts == ts and sup == sup:
                metrics["fund_stake_ratio"] = float(ts / (sup + 1e-12))

        score = grt_score(metrics, probs)

        if auto_save:
            row = {
                "as_of_date": str(df.index.max()),
                "exchange": used_ex,
                "symbol": used_sym,
                "benchmark": used_bench,
                "updated_at_utc": now_utc_iso(),
                "score_0_100": float(score),
                "p_up_7d": safe_float(probs.get(7)),
                "p_up_30d": safe_float(probs.get(30)),
                "p_up_90d": safe_float(probs.get(90)),
                **{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in metrics.items()},
            }
            save_row_to_csv(row)

        st.success(f"✅ Actualizado ({used_ex} · {used_sym}) — Score: {score:.1f}/100")
        st.caption(f"Benchmark usado: {dfb.attrs.get('exchange_used', used_ex)} · {used_bench}")

    except Exception as e:
        st.error(f"Error al actualizar: {e}")

# Si no actualizas, mostramos último guardado
hist = load_results()
if (not run_update) and (not hist.empty):
    last = hist.sort_values("as_of_date").iloc[-1].to_dict()
    score = safe_float(last.get("score_0_100"))
    probs = {7: safe_float(last.get("p_up_7d")), 30: safe_float(last.get("p_up_30d")), 90: safe_float(last.get("p_up_90d"))}

# =========================
# KPIs
# =========================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div class='kpi'><div class='label'>GRT Score</div><div class='value'>{(score if score==score else np.nan):.1f}/100</div><div class='hint'>Confluencia</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='kpi'><div class='label'>P(↑ 7 días)</div><div class='value'>{(probs.get(7)*100 if probs.get(7)==probs.get(7) else np.nan):.1f}%</div><div class='hint'>Ensemble</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='kpi'><div class='label'>P(↑ 30 días)</div><div class='value'>{(probs.get(30)*100 if probs.get(30)==probs.get(30) else np.nan):.1f}%</div><div class='hint'>Ensemble</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='kpi'><div class='label'>P(↑ 90 días)</div><div class='value'>{(probs.get(90)*100 if probs.get(90)==probs.get(90) else np.nan):.1f}%</div><div class='hint'>Ensemble</div></div>", unsafe_allow_html=True)

st.write("")

# Gauge
if score == score:
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        number={"suffix": "/100"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#A855F7"}},
        title={"text": "Confluencia (Score)"}
    ))
    fig_g.update_layout(height=260, margin=dict(l=20,r=20,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_g, use_container_width=True)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📈 Market", "🧠 Modelos", "🟣 Fundamentals GRT", "🧾 Histórico / Export"])

with tab1:
    if df.empty:
        st.info("Pulsa **Actualizar**.")
    else:
        cd = df.reset_index()
        fig_c = go.Figure(data=[go.Candlestick(
            x=cd["date"], open=cd["open"], high=cd["high"], low=cd["low"], close=cd["close"]
        )])
        fig_c.update_layout(height=420, margin=dict(l=20,r=20,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c, use_container_width=True)

with tab2:
    st.subheader("Modelos · Probabilidades + Explicabilidad")
    if not report:
        st.info("Pulsa **Actualizar**.")
    else:
        rows = []
        for h in [7,30,90]:
            rep = report.get(h, {})
            rows.append({
                "Horizonte": f"{h} días",
                "P(↑)": probs.get(h, np.nan),
                "AUC (test)": rep.get("auc", np.nan),
                "N train": rep.get("n_train", 0),
                "N test": rep.get("n_test", 0),
                "Rows clean": diag.get(h, {}).get("rows_after_clean", 0),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        h_sel = st.selectbox("Ver factores top para horizonte", [7,30,90], index=1)
        top_feats = report.get(h_sel, {}).get("top_features", [])
        if not top_feats:
            st.warning("No se pudo calcular importance (si AUC inválido o pocos datos).")
        else:
            imp_df = pd.DataFrame(top_feats, columns=["feature","importance"])
            fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h",
                             title=f"Top factores (perm. importance) — {h_sel}d")
            fig_imp.update_layout(height=420, margin=dict(l=20,r=20,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_imp, use_container_width=True)

with tab3:
    st.subheader("Fundamentals GRT · Graph Network (Gateway)")
    if fund is None or fund.empty:
        st.warning("No se pudieron cargar fundamentals. Añade THE_GRAPH_API_KEY en el sidebar y pulsa Actualizar.")
    else:
        st.dataframe(fund, use_container_width=True)

with tab4:
    st.subheader("Histórico / Export")
    if hist.empty:
        st.info("Aún no hay histórico.")
    else:
        st.dataframe(hist.sort_values("as_of_date").tail(60), use_container_width=True)
        with open(RESULTS_PATH, "rb") as f:
            st.download_button("⬇️ Descargar daily_results.csv", f, file_name="daily_results.csv", mime="text/csv")

st.caption("⚠️ Panel probabilístico. No garantiza nada. La clave para GRT es combinar precio + fundamentals de red + riesgo.")
