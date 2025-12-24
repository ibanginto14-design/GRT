# app.py
import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

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
from sklearn.inspection import permutation_importance

# HMM (opcional, si no está instalado, la app sigue)
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

# Plotly para look épico
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

GRAPH_NETWORK_SUBGRAPH = "https://api.thegraph.com/subgraphs/name/graphprotocol/graph-network-mainnet"

DEFAULT_SETTINGS = {
    "preferred_exchange": "binance",
    "symbol": "GRT/USDT",
    "benchmark": "BTC/USDT",
    "days": 900,
    "timeframe": "1d",
    "api_keys": {
        # Opcional (si quieres on-chain “hardcore” real):
        # "covalent": "TU_KEY",
        # "etherscan": "TU_KEY",
        # "bitquery": "TU_KEY",
        # "glassnode": "TU_KEY",
        # "lunarcrush": "TU_KEY",
        # "cryptopanic": "TU_KEY",
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
  --card2:#101B3B;
  --text:#E8EEFF;
  --muted:#AAB6E8;
  --accent:#A855F7;
  --accent2:#22C55E;
  --warn:#F59E0B;
  --bad:#EF4444;
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
.block-container{
  padding-top: 1.2rem;
}
.epic-hero{
  padding: 18px 18px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(168,85,247,0.20), rgba(34,197,94,0.08));
  border: 1px solid rgba(30,42,85,0.7);
  box-shadow: 0 18px 60px rgba(0,0,0,0.35);
}
.epic-title{
  font-size: 26px;
  font-weight: 800;
  letter-spacing: -0.3px;
}
.epic-sub{
  color: var(--muted);
  font-size: 14px;
}
.kpi-grid{
  display:grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-top: 12px;
}
.kpi{
  border-radius: 16px;
  padding: 14px 14px;
  background: linear-gradient(180deg, rgba(15,23,48,0.95), rgba(16,27,59,0.85));
  border: 1px solid rgba(30,42,85,0.7);
}
.kpi .label{
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 6px;
}
.kpi .value{
  font-size: 20px;
  font-weight: 800;
}
.kpi .hint{
  color: var(--muted);
  font-size: 11px;
  margin-top: 6px;
}
.badge{
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(168,85,247,0.15);
  border:1px solid rgba(168,85,247,0.35);
  color: var(--text);
  font-size: 12px;
  margin-left: 8px;
}
hr{
  border-color: rgba(30,42,85,0.6) !important;
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
def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int = 365) -> pd.DataFrame:
    """
    Descarga OHLCV con fallback anti-451/restricciones.
    Devuelve df con attrs: exchange_used, symbol_used
    """
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
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
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
    df["log_close"] = np.log(df["close"].astype(float))
    df["ret"] = df["close"].astype(float).pct_change()
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

def _graphql_post(url: str, query: str, variables: Optional[dict] = None, timeout=20) -> dict:
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=6 * 60 * 60)
def fetch_grt_network_fundamentals(days: int = 365) -> pd.DataFrame:
    """
    Métricas fundamentales de The Graph Network (GRT) desde el subgraph oficial.
    Nota: El subgraph da entidades; agregamos a una serie "diaria" aproximada usando epochs o snapshots cuando estén.
    Si el subgraph cambia, puede requerir ajustar el query.
    """
    # Estrategia pragmática:
    # - Many subgraphs expose "graphNetwork" + "indexers" + "delegators" + "pools" etc.
    # - No siempre hay "daily snapshots" consistentes.
    # - Por eso, hacemos:
    #   1) Query del estado global (staking, supply, indexers count)
    #   2) Creamos un DF "as-of" (último punto). Esto ya suma muchísimo para el score.
    # Si en tu caso quieres series diarias reales, lo ampliamos con un endpoint de "epochs" si lo vemos disponible.

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

def trend_regression(df: pd.DataFrame, window: int = 90) -> dict:
    d = df.dropna().tail(window).copy()
    y = d["log_close"].values
    x = np.arange(len(d))
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return {"trend_slope": float(model.params[1]), "trend_pvalue": float(model.pvalues[1]), "trend_r2": float(model.rsquared)}

def stationarity_tests(df: pd.DataFrame) -> dict:
    d = df["log_ret"].dropna()
    adf = adfuller(d, autolag="AIC")
    kpss_stat, kpss_p, _, _ = kpss(d, regression="c", nlags="auto")
    return {"adf_stat": float(adf[0]), "adf_pvalue": float(adf[1]), "kpss_stat": float(kpss_stat), "kpss_pvalue": float(kpss_p)}

def momentum_metrics(df: pd.DataFrame) -> dict:
    r = df["ret"].dropna()
    r30 = float((1 + r.tail(30)).prod() - 1) if len(r) >= 30 else np.nan
    r90 = float((1 + r.tail(90)).prod() - 1) if len(r) >= 90 else np.nan
    recent = r.tail(14)
    z = float((recent.mean() - r.mean()) / (r.std() + 1e-12)) if len(r) > 60 else np.nan
    return {"mom_ret_30d": r30, "mom_ret_90d": r90, "mom_z_14d": z}

def volume_signal(df: pd.DataFrame) -> dict:
    v = df["volume"].astype(float)
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
    skew = float(w.skew())
    kurt = float(w.kurt())
    downside = w[w < 0]
    downside_vol = float(downside.std()) if len(downside) > 10 else np.nan
    return {"skew_90d": skew, "kurt_90d": kurt, "downside_vol_90d": downside_vol}

def structural_breaks(df: pd.DataFrame) -> dict:
    y = df["log_close"].dropna().values
    if len(y) < 180:
        return {"breakpoints_n": np.nan, "break_recent": np.nan}
    algo = rpt.Pelt(model="rbf").fit(y)
    bkps = algo.predict(pen=8)
    # bkps incluye len(y) al final
    b = bkps[:-1]
    n = len(b)
    # si hubo ruptura en los últimos 30 días:
    recent = any((len(y) - bp) <= 30 for bp in b) if n > 0 else False
    return {"breakpoints_n": float(n), "break_recent": float(1.0 if recent else 0.0)}

def rolling_correlation(df_asset: pd.DataFrame, df_bench: pd.DataFrame, window: int = 60) -> dict:
    joined = pd.concat([df_asset["ret"].rename("asset"), df_bench["ret"].rename("bench")], axis=1).dropna()
    if len(joined) < window + 10:
        return {"corr_60d": np.nan}
    return {"corr_60d": float(joined["asset"].rolling(window).corr(joined["bench"]).iloc[-1])}

def hmm_regimes(df: pd.DataFrame) -> dict:
    """
    HMM sobre [ret, vol_rolling] para detectar regímenes.
    Devuelve: prob del régimen actual y una etiqueta interpretada.
    """
    if not HMM_AVAILABLE:
        return {"hmm_regime": None, "hmm_p_regime": np.nan}

    r = df["ret"].dropna()
    if len(r) < 400:
        return {"hmm_regime": None, "hmm_p_regime": np.nan}

    vol = r.rolling(14).std()
    x = pd.concat([r, vol], axis=1).dropna()
    X = x.values

    # Fit HMM 3 estados
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=250, random_state=7)
    model.fit(X)

    post = model.predict_proba(X)
    current_state = int(np.argmax(post[-1]))
    p_state = float(np.max(post[-1]))

    # Interpretación simple por medias de retorno de cada estado
    states = model.predict(X)
    means = []
    for s in range(3):
        means.append(float(np.mean(r.loc[x.index][states == s])))

    # etiqueta por retorno medio:
    order = np.argsort(means)  # low -> high
    # low: dump, mid: range, high: uptrend
    label_map = {int(order[0]): "Dump / Risk", int(order[1]): "Range / Accum", int(order[2]): "Uptrend"}
    label = label_map.get(current_state, "Unknown")

    return {"hmm_regime": label, "hmm_p_regime": p_state}

def build_feature_frame(df: pd.DataFrame, bench: pd.DataFrame, grt_fund: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Construye features para modelos.
    """
    d = df.copy()
    d["ret_1"] = d["ret"]
    d["ret_7"] = d["close"].pct_change(7)
    d["ret_30"] = d["close"].pct_change(30)
    d["vol_14"] = d["ret"].rolling(14).std()
    d["vol_60"] = d["ret"].rolling(60).std()
    d["mom_30"] = (1 + d["ret"]).rolling(30).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    d["vol_z_14"] = (d["volume"].rolling(14).mean() - d["volume"].rolling(180).mean()) / (d["volume"].rolling(180).std() + 1e-12)

    # benchmark
    j = pd.concat([d["ret"], bench["ret"].rename("bench_ret")], axis=1).dropna()
    d["corr_60"] = j["ret"].rolling(60).corr(j["bench_ret"])

    # Tail risk
    d["skew_90"] = d["ret"].rolling(90).skew()
    d["kurt_90"] = d["ret"].rolling(90).kurt()

    # Fundamentals GRT (as-of), lo “broadcast” a todas las filas para que el modelo pueda usarlo
    # (si tuvieras series diarias reales, aquí sería merge por fecha)
    if grt_fund is not None and not grt_fund.empty:
        f = grt_fund.iloc[-1].to_dict()
        for k, v in f.items():
            d[f"fund_{k}"] = v

        # ratios útiles
        if d.get("fund_totalTokensStaked") is not None and d.get("fund_totalSupply") is not None:
            d["fund_stake_ratio"] = d["fund_totalTokensStaked"] / (d["fund_totalSupply"] + 1e-12)

    return d

def fit_ensemble_probabilities(feat: pd.DataFrame, horizons=(7, 30, 90)) -> Tuple[dict, dict]:
    """
    Entrena modelos out-of-sample simplificado (time split) para prob de subida a H días.
    Devuelve:
      probs: {h: prob_up}
      model_report: {h: {auc, n_train, n_test, top_features}}
    """
    feat = feat.copy()

    # Feature columns
    candidates = [c for c in feat.columns if c not in ("open","high","low","close","volume","log_close","ret","log_ret")]
    # Mantén solo numéricos
    X_all = feat[candidates].apply(pd.to_numeric, errors="coerce")

    probs = {}
    report = {}

    for h in horizons:
        y = (feat["close"].shift(-h) / feat["close"] - 1.0) > 0
        y = y.astype(float)

        data = pd.concat([X_all, y.rename("y")], axis=1).dropna()
        if len(data) < 600:
            probs[h] = np.nan
            report[h] = {"auc": np.nan, "n_train": 0, "n_test": 0, "top_features": []}
            continue

        # Time split: 80% train, 20% test (último tramo)
        n = len(data)
        cut = int(n * 0.8)
        train = data.iloc[:cut]
        test = data.iloc[cut:]

        X_train = train.drop(columns=["y"])
        y_train = train["y"].astype(int)
        X_test = test.drop(columns=["y"])
        y_test = test["y"].astype(int)

        # 1) Logistic
        logit = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=500, n_jobs=1))
        ])
        # 2) Gradient Boosting
        gbt = GradientBoostingClassifier(random_state=7)

        logit.fit(X_train, y_train)
        gbt.fit(X_train, y_train)

        # Ensemble average prob
        p_logit = logit.predict_proba(X_test)[:, 1]
        p_gbt = gbt.predict_proba(X_test)[:, 1]
        p_ens = 0.5 * p_logit + 0.5 * p_gbt

        auc = roc_auc_score(y_test, p_ens) if len(np.unique(y_test)) > 1 else np.nan

        # Probabilidad HOY con últimas features disponibles
        last_row = X_all.dropna().iloc[[-1]]
        p_today = 0.5 * logit.predict_proba(last_row)[:, 1][0] + 0.5 * gbt.predict_proba(last_row)[:, 1][0]
        probs[h] = float(p_today)

        # Explainability: permutation importance sobre test con el GBT (rápido y estable)
        try:
            imp = permutation_importance(gbt, X_test, y_test, n_repeats=7, random_state=7, scoring="roc_auc")
            imp_df = pd.DataFrame({"feature": X_test.columns, "importance": imp.importances_mean})
            top = imp_df.sort_values("importance", ascending=False).head(10)
            top_feats = list(zip(top["feature"].tolist(), top["importance"].astype(float).round(4).tolist()))
        except Exception:
            top_feats = []

        report[h] = {"auc": float(auc) if auc == auc else np.nan, "n_train": int(len(train)), "n_test": int(len(test)), "top_features": top_feats}

    return probs, report

def grt_score(metrics: dict, probs: dict) -> float:
    """
    Score final 0..100 con confluencia:
    - Tendencia, momentum, volumen, volatilidad, rupturas, HMM, fundamentals, probs.
    """
    s = 50.0

    # Trend
    slope = metrics.get("trend_slope", np.nan)
    pval = metrics.get("trend_pvalue", np.nan)
    if slope == slope and pval == pval:
        s += 14 if (slope > 0 and pval < 0.05) else (7 if slope > 0 else (-14 if (slope < 0 and pval < 0.05) else (-7 if slope < 0 else 0)))

    # Momentum
    m30 = metrics.get("mom_ret_30d", np.nan)
    if m30 == m30:
        s += 10 if m30 > 0 else -10

    # Volume Z
    vz = metrics.get("vol_z_14d", np.nan)
    if vz == vz:
        s += 7 if vz > 0.5 else (-7 if vz < -0.5 else 0)

    # Volatility
    gv = metrics.get("garch_vol_now", np.nan)
    if gv == gv:
        s += 3 if gv < 4 else (-6 if gv > 9 else -2)

    # Breaks
    br = metrics.get("break_recent", np.nan)
    if br == br:
        # ruptura reciente = oportunidad o riesgo; aquí penalizamos si coincide con tail risk negativo
        s += 2 if br < 0.5 else -2

    # Tail risk
    skew = metrics.get("skew_90d", np.nan)
    if skew == skew:
        s += 4 if skew > 0.2 else (-4 if skew < -0.2 else 0)

    downside = metrics.get("downside_vol_90d", np.nan)
    if downside == downside:
        s += 2 if downside < 0.03 else (-4 if downside > 0.06 else -1)

    # Correlation (desacople)
    corr = metrics.get("corr_60d", np.nan)
    if corr == corr:
        s += 3 if corr < 0.5 else 0

    # HMM
    regime = metrics.get("hmm_regime", None)
    p_reg = metrics.get("hmm_p_regime", np.nan)
    if regime and isinstance(regime, str):
        if "Uptrend" in regime:
            s += 8 if (p_reg == p_reg and p_reg > 0.6) else 5
        elif "Range" in regime:
            s += 2
        elif "Dump" in regime:
            s -= 8

    # Fundamentals: stake ratio
    stake_ratio = metrics.get("fund_stake_ratio", np.nan)
    if stake_ratio == stake_ratio:
        # ratio alto suele ser estructuralmente positivo (menos float líquido)
        s += 6 if stake_ratio > 0.35 else (3 if stake_ratio > 0.25 else 0)

    # Ensemble probabilities
    p30 = probs.get(30, np.nan)
    p90 = probs.get(90, np.nan)
    if p30 == p30:
        s += 8 if p30 > 0.6 else (-8 if p30 < 0.45 else 0)
    if p90 == p90:
        s += 6 if p90 > 0.6 else (-6 if p90 < 0.45 else 0)

    return float(np.clip(s, 0, 100))

def regime_badge(score: float) -> Tuple[str, str]:
    if score >= 75:
        return "BULLISH", "rgba(34,197,94,0.20)"
    if score >= 58:
        return "LEAN-BULL", "rgba(34,197,94,0.10)"
    if score >= 42:
        return "NEUTRAL", "rgba(168,85,247,0.12)"
    if score >= 25:
        return "RISK-OFF", "rgba(245,158,11,0.18)"
    return "HIGH RISK", "rgba(239,68,68,0.18)"

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
# SIDEBAR CONTROLS
# =========================
settings = load_settings()

with st.sidebar:
    st.markdown("## 🟣 GRT QuantLab")
    st.caption("Panel cuantitativo + fundamentals de red. Actualizable diario.")
    st.divider()

    preferred_exchange = st.selectbox(
        "Exchange preferido (fallback auto)",
        ["binance", "kraken", "coinbase", "bitstamp"],
        index=["binance","kraken","coinbase","bitstamp"].index(settings.get("preferred_exchange","binance"))
    )
    symbol = st.text_input("Símbolo", value=settings.get("symbol","GRT/USDT"))
    benchmark = st.text_input("Benchmark", value=settings.get("benchmark","BTC/USDT"))
    days = st.slider("Histórico (días)", 250, 2000, int(settings.get("days", 900)), step=50)

    st.divider()
    st.markdown("### ⚙️ Modo")
    auto_save = st.toggle("Guardar resultado al actualizar", value=True)
    run_update = st.button("🔄 Actualizar (calcular todo)", type="primary")

    st.divider()
    st.markdown("### 🔌 APIs opcionales (on-chain/sentimiento)")
    st.caption("Para flows/holders/CDD necesitas APIs. Si no, sale N/A sin romper.")
    api_keys = settings.get("api_keys", {})
    covalent_key = st.text_input("Covalent key (opcional)", value=api_keys.get("covalent",""), type="password")
    etherscan_key = st.text_input("Etherscan key (opcional)", value=api_keys.get("etherscan",""), type="password")

    if st.button("💾 Guardar ajustes"):
        settings["preferred_exchange"] = preferred_exchange
        settings["symbol"] = symbol.strip()
        settings["benchmark"] = benchmark.strip()
        settings["days"] = int(days)
        settings.setdefault("api_keys", {})
        settings["api_keys"]["covalent"] = covalent_key.strip()
        settings["api_keys"]["etherscan"] = etherscan_key.strip()
        save_settings(settings)
        st.success("Ajustes guardados.")


# =========================
# HERO
# =========================
st.markdown(
    f"""
    <div class="epic-hero">
      <div class="epic-title">🟣 GRT QuantLab <span class="badge">Regímenes · Probabilidades · Fundamentals</span></div>
      <div class="epic-sub">
        Score por confluencia (0–100) + heatmap de probabilidades (7/30/90 días) + explicación de factores.
        Datos: CCXT (precio) + Graph Network subgraph (fundamentals).
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# =========================
# COMPUTE ALL
# =========================
def compute_all(preferred_exchange: str, symbol: str, benchmark: str, days: int) -> Tuple[dict, dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tf = "1d"
    df = compute_returns(fetch_ohlcv(preferred_exchange, symbol, tf, limit=days))
    dfb = compute_returns(fetch_ohlcv(preferred_exchange, benchmark, tf, limit=days))

    # Fundamentals GRT
    try:
        fund = fetch_grt_network_fundamentals(days=365)
    except Exception:
        fund = pd.DataFrame()

    # Core metrics
    metrics = {}
    metrics.update(trend_regression(df, 90))
    metrics.update(stationarity_tests(df))
    metrics.update(momentum_metrics(df))
    metrics.update(volume_signal(df))
    metrics.update(garch_volatility(df))
    metrics.update(structural_breaks(df))
    metrics.update(rolling_correlation(df, dfb, 60))
    metrics.update(tail_risk_metrics(df))

    # HMM
    metrics.update(hmm_regimes(df))

    # Append fundamentals (as-of)
    if not fund.empty:
        last_f = fund.iloc[-1].to_dict()
        for k, v in last_f.items():
            metrics[f"fund_{k}"] = safe_float(v)
        # ratios
        ts = metrics.get("fund_totalTokensStaked", np.nan)
        sup = metrics.get("fund_totalSupply", np.nan)
        if ts == ts and sup == sup:
            metrics["fund_stake_ratio"] = float(ts / (sup + 1e-12))

    # Models
    feat = build_feature_frame(df, dfb, fund if not fund.empty else None)
    probs, report = fit_ensemble_probabilities(feat, horizons=(7,30,90))

    # Final score
    score = grt_score(metrics, probs)
    return metrics, probs, report, df, dfb, fund, score


metrics = {}
probs = {}
report = {}
df = pd.DataFrame()
dfb = pd.DataFrame()
fund = pd.DataFrame()
score = np.nan

if run_update:
    try:
        metrics, probs, report, df, dfb, fund, score = compute_all(preferred_exchange, symbol, benchmark, days)
        used_ex = df.attrs.get("exchange_used", preferred_exchange)
        used_sym = df.attrs.get("symbol_used", symbol)
        used_bench = dfb.attrs.get("symbol_used", benchmark)

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
    except Exception as e:
        st.error(f"Error al actualizar: {e}")


# =========================
# If not updated this session, show last saved (if any)
# =========================
res_df = load_results()
if (not run_update) and (not res_df.empty):
    last = res_df.sort_values("as_of_date").iloc[-1].to_dict()
    score = safe_float(last.get("score_0_100"))
    probs = {7: safe_float(last.get("p_up_7d")), 30: safe_float(last.get("p_up_30d")), 90: safe_float(last.get("p_up_90d"))}


# =========================
# KPIs
# =========================
label, bg = regime_badge(score if score == score else 50.0)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"<div class='kpi'><div class='label'>GRT Score</div><div class='value'>{(score if score==score else np.nan):.1f}/100</div><div class='hint'>{label}</div></div>", unsafe_allow_html=True)
with kpi2:
    st.markdown(f"<div class='kpi'><div class='label'>P(↑ 7 días)</div><div class='value'>{(probs.get(7, np.nan)*100 if probs.get(7)==probs.get(7) else np.nan):.1f}%</div><div class='hint'>Ensemble</div></div>", unsafe_allow_html=True)
with kpi3:
    st.markdown(f"<div class='kpi'><div class='label'>P(↑ 30 días)</div><div class='value'>{(probs.get(30, np.nan)*100 if probs.get(30)==probs.get(30) else np.nan):.1f}%</div><div class='hint'>Ensemble</div></div>", unsafe_allow_html=True)
with kpi4:
    st.markdown(f"<div class='kpi'><div class='label'>P(↑ 90 días)</div><div class='value'>{(probs.get(90, np.nan)*100 if probs.get(90)==probs.get(90) else np.nan):.1f}%</div><div class='hint'>Ensemble</div></div>", unsafe_allow_html=True)

st.write("")

# Gauge Plotly (épico)
if score == score:
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        number={"suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#A855F7"},
            "steps": [
                {"range": [0, 25], "color": "rgba(239,68,68,0.25)"},
                {"range": [25, 42], "color": "rgba(245,158,11,0.20)"},
                {"range": [42, 58], "color": "rgba(168,85,247,0.18)"},
                {"range": [58, 75], "color": "rgba(34,197,94,0.18)"},
                {"range": [75, 100], "color": "rgba(34,197,94,0.28)"},
            ],
        },
        title={"text": "Confluencia (Score)"}
    ))
    fig_g.update_layout(height=260, margin=dict(l=20,r=20,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
else:
    fig_g = None

left, right = st.columns([1.1, 1.9])
with left:
    if fig_g:
        st.plotly_chart(fig_g, use_container_width=True)
    st.caption(f"🕒 {now_utc_iso()} · *Si Binance falla, hay fallback automático.*")

with right:
    # Heatmap de probabilidades
    hm = pd.DataFrame({
        "Horizonte": ["7 días", "30 días", "90 días"],
        "Prob. subida": [
            probs.get(7, np.nan),
            probs.get(30, np.nan),
            probs.get(90, np.nan),
        ],
        "Riesgo (proxy)": [
            safe_float(metrics.get("garch_vol_now", np.nan)),
            safe_float(metrics.get("downside_vol_90d", np.nan)),
            safe_float(metrics.get("breakpoints_n", np.nan)),
        ],
        "Régimen HMM": [
            metrics.get("hmm_regime", None),
            metrics.get("hmm_regime", None),
            metrics.get("hmm_regime", None),
        ]
    })

    fig_hm = px.imshow(
        np.array([[hm.loc[0,"Prob. subida"], hm.loc[1,"Prob. subida"], hm.loc[2,"Prob. subida"]]]),
        x=["7d", "30d", "90d"],
        y=["P(up)"],
        aspect="auto",
        color_continuous_scale="Purples",
        zmin=0, zmax=1
    )
    fig_hm.update_layout(height=260, margin=dict(l=20,r=20,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_hm, use_container_width=True)

st.divider()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Market", "🧠 Modelos", "🟣 Fundamentals GRT", "🧾 Histórico / Export"])

with tab1:
    st.subheader("Market · Precio, Volumen, Señales")
    if df is None or df.empty:
        st.info("Pulsa **Actualizar** en la barra lateral para calcular todo.")
    else:
        used_ex = df.attrs.get("exchange_used", preferred_exchange)
        used_sym = df.attrs.get("symbol_used", symbol)
        st.caption(f"Datos usados: **{used_ex}** — **{used_sym}**")

        # Candles
        cd = df.reset_index().rename(columns={"index":"date"})
        fig_c = go.Figure(data=[go.Candlestick(
            x=cd["date"], open=cd["open"], high=cd["high"], low=cd["low"], close=cd["close"],
            name="OHLC"
        )])
        fig_c.update_layout(height=420, margin=dict(l=20,r=20,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c, use_container_width=True)

        # Volume
        fig_v = go.Figure()
        fig_v.add_trace(go.Bar(x=cd["date"], y=cd["volume"], name="Volume"))
        fig_v.update_layout(height=260, margin=dict(l=20,r=20,t=20,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_v, use_container_width=True)

        # Señales principales
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Trend slope (90d)", f"{safe_float(metrics.get('trend_slope')):.4f}")
        cB.metric("Momentum 30d", f"{safe_float(metrics.get('mom_ret_30d'))*100:.2f}%")
        cC.metric("Vol z (14d)", f"{safe_float(metrics.get('vol_z_14d')):.2f}")
        cD.metric("GARCH vol", f"{safe_float(metrics.get('garch_vol_now')):.2f}")

        st.write("")
        st.subheader("Regímenes (HMM)")
        if not HMM_AVAILABLE:
            st.warning("HMM no está disponible (instala `hmmlearn`). La app sigue funcionando sin HMM.")
        else:
            st.write(f"**Régimen actual:** {metrics.get('hmm_regime')} · **Confianza:** {safe_float(metrics.get('hmm_p_regime')):.2f}")

with tab2:
    st.subheader("Modelos · Probabilidades + Explicabilidad")
    st.caption("Modelos: Logistic + GradientBoosting (ensemble). Validación temporal simple (train 80% / test 20%).")

    if not report:
        st.info("Pulsa **Actualizar** para entrenar y generar probabilidades.")
    else:
        # Report table
        rows = []
        for h in [7,30,90]:
            rep = report.get(h, {})
            rows.append({
                "Horizonte": f"{h} días",
                "P(↑)": probs.get(h, np.nan),
                "AUC (test)": rep.get("auc", np.nan),
                "N train": rep.get("n_train", 0),
                "N test": rep.get("n_test", 0),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Explainability: top features (30d por defecto)
        st.write("")
        h_sel = st.selectbox("Ver factores top para horizonte", [7,30,90], index=1)
        top_feats = report.get(h_sel, {}).get("top_features", [])
        if not top_feats:
            st.warning("No se pudo calcular importance (puede pasar si hay pocos datos o AUC inválido).")
        else:
            imp_df = pd.DataFrame(top_feats, columns=["feature","importance"])
            fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h", title=f"Top factores (perm. importance) — {h_sel}d")
            fig_imp.update_layout(height=420, margin=dict(l=20,r=20,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_imp, use_container_width=True)

with tab3:
    st.subheader("Fundamentals GRT · Graph Network")
    st.caption("Fuente: Graph Network subgraph. Esto da ventaja real en GRT (adopción / staking / estructura).")

    if fund is None or fund.empty:
        st.warning("No se pudieron cargar fundamentals (o subgraph no respondió). Pulsa Actualizar.")
    else:
        st.dataframe(fund, use_container_width=True)

        # KPIs
        f = fund.iloc[-1].to_dict()
        stake = safe_float(f.get("totalTokensStaked"))
        supply = safe_float(f.get("totalSupply"))
        stake_ratio = (stake / (supply + 1e-12)) if (stake==stake and supply==supply) else np.nan

        a,b,c,d = st.columns(4)
        a.metric("Total staked", f"{stake:,.0f}" if stake==stake else "NA")
        b.metric("Stake ratio", f"{stake_ratio*100:.2f}%" if stake_ratio==stake_ratio else "NA")
        c.metric("Active indexers", f"{safe_float(f.get('activeIndexers')):.0f}" if safe_float(f.get('activeIndexers'))==safe_float(f.get('activeIndexers')) else "NA")
        d.metric("Delegators (approx)", f"{safe_float(f.get('delegatorsApprox')):.0f}" if safe_float(f.get('delegatorsApprox'))==safe_float(f.get('delegatorsApprox')) else "NA")

        st.write("")
        st.info("Siguiente nivel (si quieres): convertir estas métricas en **series diarias** consultando snapshots/epochs del subgraph. La app está preparada para ello.")

with tab4:
    st.subheader("Histórico / Export")
    hist = load_results()
    if hist.empty:
        st.info("Aún no hay histórico. Pulsa **Actualizar** con 'Guardar' activado.")
    else:
        st.dataframe(hist.sort_values("as_of_date").tail(60), use_container_width=True)
        with open(RESULTS_PATH, "rb") as f:
            st.download_button("⬇️ Descargar daily_results.csv", f, file_name="daily_results.csv", mime="text/csv")

st.divider()

# Footer hints
st.caption("⚠️ Esto no “garantiza” subidas. Optimiza decisión con confluencia + gestión de riesgo. Para on-chain tipo flows/holders/CDD necesitarás APIs con datos etiquetados.")
