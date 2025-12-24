import os
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

import requests
import ccxt
import xml.etree.ElementTree as ET

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import coint, grangercausalitytests

from arch import arch_model
import ruptures as rpt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

import plotly.graph_objects as go
import plotly.express as px

from email.utils import parsedate_to_datetime

# HMM opcional
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

# EVT opcional (SciPy)
try:
    from scipy.stats import genpareto
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="GRT QuantLab", page_icon="🟣", layout="wide")

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
    "api_keys": {
        "thegraph_gateway": ""   # opcional
    },
    "news": {
        "enable": True,
        "lookback_days": 14,
        "rss_timeout": 15
    },
    "backtest": {
        "horizon_days": 30,
        "step_days": 7,
        "min_train_rows": 320,
        "thr_long": 0.60,
        "thr_cash": 0.45,
        "fee_bps": 8,  # 0.08% por cambio de posición (aprox)
    }
}


# =========================
# UI THEME (épico + micro-animaciones)
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

.block-container{ padding-top: 1.1rem; }

.epic-hero{
  padding: 18px 18px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(168,85,247,0.20), rgba(34,197,94,0.08));
  border: 1px solid rgba(30,42,85,0.7);
  box-shadow: 0 18px 60px rgba(0,0,0,0.35);
}

.epic-title{ font-size: 26px; font-weight: 900; letter-spacing: -0.4px; }
.epic-sub{ color: var(--muted); font-size: 14px; margin-top: 6px; }

.badge{
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  margin-left: 10px;
  font-size: 12px;
  color: var(--text);
  border: 1px solid rgba(168,85,247,0.35);
  background: rgba(168,85,247,0.12);
}

.kpi{
  border-radius: 16px;
  padding: 14px 14px;
  background: linear-gradient(180deg, rgba(15,23,48,0.95), rgba(16,27,59,0.85));
  border: 1px solid rgba(30,42,85,0.7);
  box-shadow: 0 14px 40px rgba(0,0,0,0.25);
  transition: transform .12s ease, border-color .12s ease;
  color: var(--text) !important;
}

.kpi:hover{
  transform: translateY(-2px);
  border-color: rgba(168,85,247,0.55);
}

.kpi *{ color: var(--text) !important; }

.kpi .label{
  color: var(--muted) !important;
  font-size: 12px;
  margin-bottom: 6px;
}

.kpi .value{
  color: var(--text) !important;
  font-size: 20px;
  font-weight: 900;
  letter-spacing: -0.2px;
}

.kpi .hint{
  color: var(--muted) !important;
  font-size: 11px;
  margin-top: 6px;
  line-height: 1.25rem;
}

.small-muted{
  color: var(--muted);
  font-size: 12px;
}

hr { border-color: rgba(30,42,85,0.55) !important; }

</style>
"""
st.markdown(EPIC_CSS, unsafe_allow_html=True)


# =========================
# HELPERS
# =========================
def _fmt_pct(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return f"{p*100:.1f}%"

def _fmt_num(x: float, digits=3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"

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

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _clean_text(s: str) -> str:
    if not s:
        return ""
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s)

def load_settings() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2, ensure_ascii=False)
        return DEFAULT_SETTINGS
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        s = json.load(f)
    # backfill keys
    def _backfill(dst, src):
        for k, v in src.items():
            if k not in dst:
                dst[k] = v
            else:
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _backfill(dst[k], v)
    _backfill(s, DEFAULT_SETTINGS)
    return s

def save_settings(s: dict) -> None:
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)

def _symbol_variants(symbol: str):
    base, quote = symbol.split("/")
    variants = [symbol]
    if quote.upper() == "USDT":
        variants += [f"{base}/USD", f"{base}/USDC"]
    return variants

def explain_score(score: float) -> str:
    if score is None or not np.isfinite(score):
        return "Sin cálculo aún. Pulsa “Actualizar”."
    if score >= 75:
        return "Confluencia alta: muchas señales alineadas. Aun así, cripto puede sorprender: gestiona riesgo."
    if score >= 60:
        return "Confluencia moderada: señales a favor, pero con dudas. Mejor confirmar con varias pruebas."
    if score >= 45:
        return "Zona neutra: señales mezcladas. Normalmente es zona de espera hasta que algo se decante."
    if score >= 30:
        return "Confluencia baja: predominan señales débiles o riesgo alto. Cautela."
    return "Riesgo elevado: el conjunto apunta más a proteger capital que a buscar subida."

def explain_auc(auc: float) -> str:
    if auc is None or not np.isfinite(auc):
        return "AUC no disponible (pocos datos o test con una sola clase)."
    if auc >= 0.70:
        return "Muy buena: el modelo separa subidas/bajadas bastante bien (en su test)."
    if auc >= 0.60:
        return "Aceptable: algo de ventaja frente al azar."
    if auc >= 0.55:
        return "Débil: pequeña ventaja, úsalo con prudencia."
    if auc >= 0.50:
        return "Casi azar: apenas discrimina."
    return "Peor que azar: podría estar invertido o sobreajustado."

def explain_prob(p: float, horizon_days: int) -> str:
    if p is None or not np.isfinite(p):
        return f"Sin probabilidad calculada para {horizon_days}d (faltan datos o el modelo no entrenó)."
    if p >= 0.75:
        return f"Alta probabilidad de subida a {horizon_days} días según el modelo (no es garantía)."
    if p >= 0.60:
        return f"Ventaja ligera a favor de subida a {horizon_days} días."
    if p >= 0.50:
        return f"Escenario equilibrado a {horizon_days} días (casi 50/50)."
    if p >= 0.40:
        return f"Ventaja ligera a favor de bajada/lateralidad a {horizon_days} días."
    return f"Probabilidad baja de subida a {horizon_days} días (más riesgo)."

def explain_sample_quality(n_train: int, n_test: int, rows_clean: int) -> str:
    n = (n_train or 0) + (n_test or 0)
    if n == 0 or rows_clean == 0:
        return "Sin datos útiles tras limpiar (NaNs/alineación)."
    if n < 250:
        return "Pocos datos: fiabilidad baja. Úsalo como referencia, no como piloto automático."
    if n < 500:
        return "Datos moderados: fiabilidad media; confirma con otras señales."
    return "Buen tamaño de muestra: fiabilidad mejor (aun con incertidumbre)."

def model_summary_text(p: float, auc: float, n_train: int, n_test: int, rows_clean: int, h: int) -> str:
    p_txt = explain_prob(p, h)
    auc_txt = explain_auc(auc)
    samp_txt = explain_sample_quality(n_train, n_test, rows_clean)

    if (p is not None and np.isfinite(p)) and (auc is not None and np.isfinite(auc)):
        if auc >= 0.62 and p >= 0.62:
            headline = "✅ Señal coherente (prob + calidad aceptable)."
        elif auc < 0.55 and p >= 0.62:
            headline = "⚠️ Probabilidad alta, pero calidad débil (ojo con sobreconfianza)."
        elif auc >= 0.62 and p < 0.50:
            headline = "⚠️ Calidad ok, pero sesgo bajista/lateral."
        else:
            headline = "🟡 Señal mixta (no concluyente)."
    else:
        headline = "—"

    return f"{headline} {p_txt} {auc_txt} {samp_txt}"

def robust_z(x: pd.Series) -> pd.Series:
    s = x.astype(float).copy()
    med = s.median()
    mad = (s - med).abs().median()
    return (s - med) / (1.4826 * mad + 1e-12)

def _parse_pubdate_to_utc_date(s: str) -> Optional[pd.Timestamp]:
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return pd.Timestamp(dt.astimezone(timezone.utc)).normalize()
    except Exception:
        return None

# ✅ FIX: helper robusto para evitar tz-aware vs tz-naive en merges
def _to_date_index_no_tz(idx) -> pd.DatetimeIndex:
    """
    Convierte un índice a DatetimeIndex 'limpio':
      - Si trae tz, lo pasa a UTC y le quita la tz
      - Si no trae tz, lo deja sin tz
      - Normaliza a medianoche (para que encaje día con día)
    """
    dti = pd.to_datetime(idx, errors="coerce")
    if isinstance(dti, pd.DatetimeIndex) and dti.tz is not None:
        dti = dti.tz_convert("UTC").tz_localize(None)
    else:
        try:
            dti = dti.tz_localize(None)
        except Exception:
            pass
    return dti.normalize()


# =========================
# DATA FETCH
# =========================
@st.cache_data(ttl=60*60)
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


# =========================
# FUNDAMENTALS (CoinGecko + Gateway)
# =========================
@st.cache_data(ttl=6*60*60)
def fetch_grt_fundamentals_coingecko() -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/the-graph"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    d = r.json()

    m = d.get("market_data", {})
    row = {
        "as_of": str(pd.Timestamp.utcnow().date()),
        "cg_price_usd": safe_float(m.get("current_price", {}).get("usd")),
        "cg_marketcap_usd": safe_float(m.get("market_cap", {}).get("usd")),
        "cg_volume_24h_usd": safe_float(m.get("total_volume", {}).get("usd")),
        "cg_circulating_supply": safe_float(m.get("circulating_supply")),
        "cg_total_supply": safe_float(m.get("total_supply")),
        "cg_max_supply": safe_float(m.get("max_supply")),
        "cg_price_change_24h_pct": safe_float(m.get("price_change_percentage_24h")),
        "cg_price_change_7d_pct": safe_float(m.get("price_change_percentage_7d")),
        "cg_price_change_30d_pct": safe_float(m.get("price_change_percentage_30d")),
    }
    return pd.DataFrame([row]).set_index("as_of")

def _graphql_post(url: str, query: str, variables=None, timeout=25) -> dict:
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=6*60*60)
def fetch_grt_network_fundamentals_gateway(thegraph_api_key: str) -> pd.DataFrame:
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
      }
      indexers(first: 1000, where: {active: true}) { id }
    }
    """
    data = _graphql_post(url, q)
    if "errors" in data:
        raise RuntimeError(f"Graph gateway error: {data['errors']}")

    d = data["data"]
    gn = d.get("graphNetwork") or {}
    idx_count = len(d.get("indexers") or [])

    row = {
        "as_of": str(pd.Timestamp.utcnow().date()),
        "totalTokensStaked": safe_float(gn.get("totalTokensStaked")),
        "totalTokensAllocated": safe_float(gn.get("totalTokensAllocated")),
        "totalDelegatedTokens": safe_float(gn.get("totalDelegatedTokens")),
        "totalSupply": safe_float(gn.get("totalSupply")),
        "activeIndexers": float(idx_count),
    }
    return pd.DataFrame([row]).set_index("as_of")


# =========================
# NEWS (RSS) -> Sentiment 0..100 + serie diaria (ventana)
# =========================
POS_WORDS = {
    "upgrade","partnership","launch","released","adoption","growth","record","surge","bull",
    "win","success","breakthrough","milestone","approval","support","integrates","expands",
    "strong","beats","positive","accumulate","accumulation","listing","listed"
}
NEG_WORDS = {
    "hack","exploit","lawsuit","ban","down","dump","bear","crash","collapse","fraud","scam",
    "risk","warning","investigation","liquidation","delay","weak","negative","sec","rejected",
    "outage","attack","security","breach","concern"
}

def score_sentiment(text: str) -> float:
    t = _clean_text(text)
    if not t.strip():
        return 0.0
    words = [w for w in t.split() if len(w) > 2]
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    raw = (pos - neg) / max(1, (pos + neg))
    return float(raw)

@st.cache_data(ttl=60*30)
def fetch_rss_items(url: str, timeout: int = 15, max_items: int = 25) -> List[dict]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    root = ET.fromstring(r.text)

    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        desc = (item.findtext("description") or "").strip()
        items.append({"title": title, "link": link, "pubDate": pub, "desc": desc})
        if len(items) >= max_items:
            break
    return items

def build_news_panel(enable: bool, lookback_days: int, timeout: int) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - score_0_100: float
      - df_news: items (hasta 50)
      - df_daily: serie diaria score 0..100 (solo ventana)
    """
    if not enable:
        return np.nan, pd.DataFrame(), pd.DataFrame()

    queries = [
        "The+Graph+GRT",
        "The+Graph+protocol",
        "GRT+token",
    ]
    rss_urls = [f"https://news.google.com/rss/search?q={q}&hl=en&gl=US&ceid=US:en" for q in queries]

    all_items = []
    for u in rss_urls:
        try:
            all_items.extend(fetch_rss_items(u, timeout=timeout, max_items=25))
        except Exception:
            continue

    if not all_items:
        return np.nan, pd.DataFrame(), pd.DataFrame()

    cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=int(lookback_days))

    rows = []
    for it in all_items:
        dt = _parse_pubdate_to_utc_date(it.get("pubDate",""))
        if dt is None:
            continue
        if dt < cutoff:
            continue
        txt = f"{it.get('title','')} {it.get('desc','')}"
        s = score_sentiment(txt)
        rows.append({
            "date": dt,
            "title": it.get("title",""),
            "pubDate": it.get("pubDate",""),
            "link": it.get("link",""),
            "sent": float(s)
        })

    df_news = pd.DataFrame(rows).drop_duplicates(subset=["title"]).head(80)
    if df_news.empty:
        return np.nan, df_news, pd.DataFrame()

    svals = df_news["sent"].astype(float).values
    svals = svals[np.isfinite(svals)]
    if len(svals) == 0:
        return np.nan, df_news, pd.DataFrame()

    svals_sorted = np.sort(svals)
    k = max(0, int(0.1 * len(svals_sorted)))
    core = svals_sorted[k:len(svals_sorted)-k] if len(svals_sorted) > 10 else svals_sorted
    m = float(np.mean(core)) if len(core) else float(np.mean(svals_sorted))

    score_0_100 = float(np.clip(50 + 50*m, 0, 100))
    df_news["sent_0_100"] = (50 + 50*df_news["sent"]).clip(0, 100)

    # Daily series (mean)
    df_daily = (df_news.groupby("date")["sent"].mean()
                .rename("sent_mean")
                .to_frame())
    df_daily["score_0_100"] = (50 + 50*df_daily["sent_mean"]).clip(0, 100)

    return score_0_100, df_news, df_daily

def explain_news_score(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "No disponible (RSS falló o está desactivado)."
    if x >= 65:
        return "Titulares claramente positivos: puede acompañar impulso, pero también puede ser hype."
    if x >= 55:
        return "Ligero sesgo positivo: apoyo contextual, no suficiente por sí solo."
    if x >= 45:
        return "Neutral: noticias mezcladas o sin señal clara."
    if x >= 35:
        return "Sesgo negativo: aumenta la probabilidad de volatilidad/venta defensiva."
    return "Muy negativo: cuidado con shocks (aunque a veces hay rebotes técnicos)."


# =========================
# TECHNICAL INDICATORS
# =========================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m_fast = ema(series, fast)
    m_slow = ema(series, slow)
    line = m_fast - m_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def bollinger_pctb(series: pd.Series, window=20, nstd=2.0) -> pd.Series:
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + nstd * sd
    lower = ma - nstd * sd
    return (series - lower) / (upper - lower + 1e-12)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr_ = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / (atr_ + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / (atr_ + 1e-12))
    dx = (100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12))
    return dx.rolling(period).mean()

def max_drawdown(close: pd.Series) -> float:
    c = close.astype(float)
    roll_max = c.cummax()
    dd = (c / roll_max) - 1.0
    return float(dd.min()) if len(dd.dropna()) else np.nan

def drawdown_series(close: pd.Series) -> pd.Series:
    c = close.astype(float)
    roll_max = c.cummax()
    return (c / roll_max) - 1.0

def drawdown_duration(close: pd.Series) -> Tuple[float, float]:
    dd = drawdown_series(close).fillna(0.0)
    in_dd = dd < 0
    durations = []
    cur = 0
    for v in in_dd.values:
        if v:
            cur += 1
        else:
            if cur > 0:
                durations.append(cur)
            cur = 0
    if cur > 0:
        durations.append(cur)
    if not durations:
        return np.nan, np.nan
    return float(np.max(durations)), float(np.mean(durations))

def time_to_recovery(close: pd.Series) -> float:
    c = close.astype(float).dropna()
    if c.empty:
        return np.nan
    dd = drawdown_series(c)
    if dd.iloc[-1] >= 0:
        return 0.0
    zeros = dd[dd >= -1e-12]
    if zeros.empty:
        return float(len(dd))
    last_zero = zeros.index[-1]
    try:
        return float((pd.Timestamp(c.index[-1]) - pd.Timestamp(last_zero)).days)
    except Exception:
        return float(len(dd) - len(zeros))

def var_cvar(r: pd.Series, alpha=0.05) -> Tuple[float, float]:
    x = r.dropna().astype(float)
    if len(x) < 250:
        return np.nan, np.nan
    q = np.quantile(x, alpha)
    cvar = float(x[x <= q].mean()) if np.any(x <= q) else np.nan
    return float(q), float(cvar)

def sharpe_simple(r: pd.Series, periods_per_year=365) -> float:
    x = r.dropna().astype(float)
    if len(x) < 180:
        return np.nan
    mu = x.mean() * periods_per_year
    sd = x.std() * np.sqrt(periods_per_year)
    return float(mu / (sd + 1e-12))

def beta_vs_bench(asset_r: pd.Series, bench_r: pd.Series, window=180) -> float:
    tmp = pd.concat([asset_r.rename("a"), bench_r.rename("b")], axis=1).dropna()
    tmp = tmp.tail(window)
    if len(tmp) < 60:
        return np.nan
    cov = np.cov(tmp["a"], tmp["b"])[0, 1]
    varb = np.var(tmp["b"])
    return float(cov / (varb + 1e-12))

def realized_vol(r: pd.Series, window=30) -> float:
    x = r.dropna().tail(window)
    if len(x) < 20:
        return np.nan
    return float(x.std() * np.sqrt(365))

def jump_stats(r: pd.Series, window=30, k=3.0) -> dict:
    x = r.dropna().astype(float)
    if len(x) < window + 50:
        return {"jump_rate": np.nan, "jumps_last90": np.nan}
    vol = x.rolling(window).std()
    jumps = (x.abs() > (k * vol)).astype(int)
    return {
        "jump_rate": float(jumps.mean()),
        "jumps_last90": float(jumps.tail(90).sum())
    }

def evt_tail_risk(r: pd.Series, alpha=0.01) -> dict:
    if not SCIPY_AVAILABLE:
        return {"evt_var_99": np.nan, "evt_es_99": np.nan, "evt_status": "SciPy not available"}
    x = r.dropna().astype(float)
    if len(x) < 500:
        return {"evt_var_99": np.nan, "evt_es_99": np.nan, "evt_status": "Not enough data"}
    losses = (-x).clip(lower=0)
    u = np.quantile(losses[losses > 0], 0.95) if np.any(losses > 0) else np.nan
    if not np.isfinite(u) or u <= 0:
        return {"evt_var_99": np.nan, "evt_es_99": np.nan, "evt_status": "No losses / invalid threshold"}
    exc = losses[losses > u] - u
    if len(exc) < 50:
        return {"evt_var_99": np.nan, "evt_es_99": np.nan, "evt_status": "Too few exceedances"}
    c, loc, scale = genpareto.fit(exc.values, floc=0.0)
    n = len(losses)
    nu = len(exc)
    pu = nu / n
    if pu <= 0 or alpha <= 0 or alpha >= pu:
        return {"evt_var_99": np.nan, "evt_es_99": np.nan, "evt_status": "alpha not in tail region"}
    q = 1 - (alpha / pu)
    if abs(c) < 1e-8:
        inv = scale * (-np.log(1 - q))
    else:
        inv = (scale / c) * ((1 - q) ** (-c) - 1)
    var = u + inv
    if c >= 1:
        es = np.nan
    else:
        es = var + (scale + c * (var - u)) / (1 - c)
    return {
        "evt_var_99": float(-var),
        "evt_es_99": float(-es) if np.isfinite(es) else np.nan,
        "evt_status": "OK"
    }


# =========================
# METRICS (base + añadidas)
# =========================
def trend_regression(df: pd.DataFrame, window: int = 90) -> dict:
    d = df.dropna().tail(window).copy()
    if len(d) < 30:
        return {"trend_slope": np.nan, "trend_pvalue": np.nan, "trend_r2": np.nan}
    y = d["log_close"].values
    x = np.arange(len(d))
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return {"trend_slope": float(model.params[1]), "trend_pvalue": float(model.pvalues[1]), "trend_r2": float(model.rsquared)}

def stationarity_tests(df: pd.DataFrame) -> dict:
    d = df["log_ret"].dropna()
    if len(d) < 120:
        return {"adf_pvalue": np.nan, "kpss_pvalue": np.nan}
    adf_res = adfuller(d, autolag="AIC")
    kpss_stat, kpss_p, _, _ = kpss(d, regression="c", nlags="auto")
    return {"adf_pvalue": float(adf_res[1]), "kpss_pvalue": float(kpss_p)}

def momentum_metrics(df: pd.DataFrame) -> dict:
    r = df["ret"].dropna()
    r7 = float((1 + r.tail(7)).prod() - 1) if len(r) >= 7 else np.nan
    r30 = float((1 + r.tail(30)).prod() - 1) if len(r) >= 30 else np.nan
    r90 = float((1 + r.tail(90)).prod() - 1) if len(r) >= 90 else np.nan
    return {"mom_ret_7d": r7, "mom_ret_30d": r30, "mom_ret_90d": r90}

def volume_signal(df: pd.DataFrame) -> dict:
    v = df["volume"]
    hist = v.tail(180)
    if len(hist) < 60:
        return {"vol_z_14d": np.nan}
    recent = v.tail(14).mean()
    z = float((recent - hist.mean()) / (hist.std() + 1e-12))
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
        "downside_vol_90d": float(downside.std()) if len(downside) > 10 else np.nan
    }

def structural_breaks(df: pd.DataFrame) -> dict:
    y = df["log_close"].dropna().values
    if len(y) < 180:
        return {"breakpoints_n": np.nan, "break_recent": np.nan}
    algo = rpt.Pelt(model="rbf").fit(y)
    bkps = algo.predict(pen=8)
    b = bkps[:-1]
    recent = any((len(y) - bp) <= 30 for bp in b) if len(b) else False
    return {"breakpoints_n": float(len(b)), "break_recent": float(1.0 if recent else 0.0)}

def extra_market_stats(df: pd.DataFrame, dfb: pd.DataFrame) -> dict:
    close = df["close"].astype(float)
    r = df["ret"].dropna().astype(float)

    dd = max_drawdown(close)
    v, c = var_cvar(r, alpha=0.05)
    sh = sharpe_simple(r)
    b = beta_vs_bench(df["ret"], dfb["ret"], window=180) if (dfb is not None and not dfb.empty) else np.nan
    rv = realized_vol(r, window=30)

    rsi14 = float(rsi(close, 14).iloc[-1]) if len(close) > 40 else np.nan
    m_line, m_sig, m_hist = macd(close)
    macd_line = float(m_line.iloc[-1]) if len(close) > 40 else np.nan
    macd_hist = float(m_hist.iloc[-1]) if len(close) > 40 else np.nan
    bb = float(bollinger_pctb(close).iloc[-1]) if len(close) > 40 else np.nan
    atr14 = float(atr(df, 14).iloc[-1]) if len(df) > 40 else np.nan
    adx14 = float(adx(df, 14).iloc[-1]) if len(df) > 60 else np.nan

    ann_ret = float((1 + r.tail(365)).prod() - 1) if len(r) >= 365 else np.nan
    ann_vol = float(r.tail(365).std() * np.sqrt(365)) if len(r) >= 180 else np.nan

    dd_max_dur, dd_avg_dur = drawdown_duration(close)

    js = jump_stats(r, window=30, k=3.0)
    evt = evt_tail_risk(r, alpha=0.01)

    return {
        "max_drawdown": dd,
        "var_95": v,
        "cvar_95": c,
        "sharpe_simple": sh,
        "beta_180": b,
        "realized_vol_30": rv,
        "rsi_14": rsi14,
        "macd_line": macd_line,
        "macd_hist": macd_hist,
        "bb_pctb": bb,
        "atr_14": atr14,
        "adx_14": adx14,
        "ann_return_est": ann_ret,
        "ann_vol_est": ann_vol,
        "dd_max_duration": dd_max_dur,
        "dd_avg_duration": dd_avg_dur,
        "jump_rate": js.get("jump_rate", np.nan),
        "jumps_last90": js.get("jumps_last90", np.nan),
        "evt_var_99": evt.get("evt_var_99", np.nan),
        "evt_es_99": evt.get("evt_es_99", np.nan),
        "evt_status": evt.get("evt_status", "N/A"),
    }

def bench_relationship_tests(df: pd.DataFrame, dfb: pd.DataFrame) -> dict:
    if df is None or df.empty or dfb is None or dfb.empty:
        return {"coint_pvalue": np.nan, "granger_min_pvalue": np.nan}

    a = pd.Series(df["close"].astype(float), index=pd.to_datetime(df.index))
    b = pd.Series(dfb["close"].astype(float), index=pd.to_datetime(dfb.index))
    tmp = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(tmp) < 300:
        return {"coint_pvalue": np.nan, "granger_min_pvalue": np.nan}

    try:
        coint_t, pval, _ = coint(np.log(tmp["a"]), np.log(tmp["b"]))
        coint_p = float(pval)
    except Exception:
        coint_p = np.nan

    ra = tmp["a"].pct_change().dropna()
    rb = tmp["b"].pct_change().dropna()
    d = pd.concat([ra.rename("asset"), rb.rename("bench")], axis=1).dropna()
    if len(d) < 400:
        gr_p = np.nan
    else:
        try:
            res = grangercausalitytests(d[["asset", "bench"]], maxlag=5, verbose=False)
            pvals = []
            for lag, out in res.items():
                pvals.append(out[0]["ssr_ftest"][1])
            gr_p = float(np.min(pvals)) if pvals else np.nan
        except Exception:
            gr_p = np.nan

    return {"coint_pvalue": coint_p, "granger_min_pvalue": gr_p}


# =========================
# HMM (ROBUSTO)
# =========================
def hmm_regimes_robust(df: pd.DataFrame) -> dict:
    if not HMM_AVAILABLE:
        return {"hmm_regime": None, "hmm_p_regime": np.nan, "hmm_status": "HMM not installed"}

    r = df["ret"].dropna()
    if len(r) < 250:
        return {"hmm_regime": None, "hmm_p_regime": np.nan, "hmm_status": "Not enough data"}

    vol = r.rolling(14).std()
    X = pd.concat([r.rename("ret"), vol.rename("vol")], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    if len(X) < 200:
        return {"hmm_regime": None, "hmm_p_regime": np.nan, "hmm_status": "Not enough clean rows"}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    Xs = Xs + np.random.normal(0, 1e-8, size=Xs.shape)

    try:
        model = GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=300,
            random_state=7
        )
        model.fit(Xs)

        post = model.predict_proba(Xs)
        current_state = int(np.argmax(post[-1]))
        p_state = float(np.max(post[-1]))

        states = model.predict(Xs)
        means = []
        r_aligned = X["ret"].values
        for s in range(3):
            means.append(float(np.mean(r_aligned[states == s])))

        order = np.argsort(means)
        label_map = {int(order[0]): "Dump / Risk", int(order[1]): "Range / Accum", int(order[2]): "Uptrend"}
        label = label_map.get(current_state, "Unknown")

        return {"hmm_regime": label, "hmm_p_regime": p_state, "hmm_status": "OK"}

    except Exception as e:
        return {"hmm_regime": None, "hmm_p_regime": np.nan, "hmm_status": f"HMM failed: {e}"}


# =========================
# FEATURES + MODELOS (incluye NEWS + fundamentals broadcast)
# =========================
def build_feature_frame(
    df: pd.DataFrame,
    bench: pd.DataFrame,
    fund_last: Optional[pd.Series],
    news_score_0_100: Optional[float]
) -> pd.DataFrame:
    d = df.copy()

    d["ret_1"] = d["ret"]
    d["ret_7"] = d["close"].pct_change(7)
    d["ret_30"] = d["close"].pct_change(30)
    d["vol_14"] = d["ret"].rolling(14).std()
    d["vol_60"] = d["ret"].rolling(60).std()
    d["mom_30"] = (1 + d["ret"]).rolling(30).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    d["vol_z_14"] = (d["volume"].rolling(14).mean() - d["volume"].rolling(180).mean()) / (d["volume"].rolling(180).std() + 1e-12)

    tmp = pd.concat([d["ret"].rename("asset"), bench["ret"].rename("bench")], axis=1)
    d["corr_60"] = tmp["asset"].rolling(60).corr(tmp["bench"])

    d["skew_90"] = d["ret"].rolling(90).skew()
    d["kurt_90"] = d["ret"].rolling(90).kurt()

    d["rsi_14"] = rsi(d["close"].astype(float), 14)
    macd_line, macd_sig, macd_hist = macd(d["close"].astype(float))
    d["macd_line"] = macd_line
    d["macd_hist"] = macd_hist
    d["bb_pctb"] = bollinger_pctb(d["close"].astype(float))
    d["atr_14"] = atr(d, 14)
    d["adx_14"] = adx(d, 14)

    if news_score_0_100 is not None and np.isfinite(news_score_0_100):
        d["news_score_0_100"] = float(news_score_0_100)
    else:
        d["news_score_0_100"] = np.nan

    if fund_last is not None and not fund_last.empty:
        for k, v in fund_last.to_dict().items():
            d[f"fund_{k}"] = safe_float(v)

        if "fund_cg_marketcap_usd" in d.columns and "fund_cg_volume_24h_usd" in d.columns:
            d["fund_mcap_to_vol"] = d["fund_cg_marketcap_usd"] / (d["fund_cg_volume_24h_usd"] + 1e-12)

        if "fund_totalTokensStaked" in d.columns and "fund_totalSupply" in d.columns:
            d["fund_stake_ratio"] = d["fund_totalTokensStaked"] / (d["fund_totalSupply"] + 1e-12)

        if "fund_cg_circulating_supply" in d.columns and "fund_cg_total_supply" in d.columns:
            d["fund_circ_ratio"] = d["fund_cg_circulating_supply"] / (d["fund_cg_total_supply"] + 1e-12)

    return d


def fit_ensemble_probabilities(feat: pd.DataFrame, horizons=(7, 30, 90), min_rows=250) -> Tuple[dict, dict, dict]:
    feat = feat.copy()

    drop_cols = {"open","high","low","close","volume","log_close","ret","log_ret"}
    candidates = [c for c in feat.columns if c not in drop_cols]

    X_all = feat[candidates].apply(pd.to_numeric, errors="coerce")
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    X_all = X_all.ffill().bfill()

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
            ("clf", LogisticRegression(max_iter=700, n_jobs=1))
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

        try:
            imp = permutation_importance(gbt, X_test, y_test, n_repeats=7, random_state=7, scoring="roc_auc")
            imp_df = pd.DataFrame({"feature": X_test.columns, "importance": imp.importances_mean})
            top = imp_df.sort_values("importance", ascending=False).head(10)
            top_feats = list(zip(top["feature"].tolist(), top["importance"].astype(float).round(4).tolist()))
        except Exception:
            top_feats = []

        report[h] = {
            "auc": float(auc) if auc == auc else np.nan,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "top_features": top_feats
        }

    return probs, report, diag


# =========================
# WALK-FORWARD BACKTEST + CALIBRATION + LIFT + CFR
# =========================
def _prep_xy_for_horizon(feat: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = {"open","high","low","close","volume","log_close","ret","log_ret"}
    candidates = [c for c in feat.columns if c not in drop_cols]
    X = feat[candidates].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill()
    y = (((feat["close"].shift(-horizon) / feat["close"]) - 1.0) > 0).astype(int)
    return X, y

@st.cache_data(ttl=60*60)
def walk_forward_prob_series(feat: pd.DataFrame, horizon: int, step_days: int = 7, min_train_rows: int = 320) -> pd.Series:
    feat = feat.copy()
    X, y = _prep_xy_for_horizon(feat, horizon)
    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    if len(data) < (min_train_rows + 50):
        return pd.Series(dtype=float)

    idx = data.index
    probs = pd.Series(index=idx, dtype=float)

    def _make_models():
        logit = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=700, n_jobs=1))
        ])
        gbt = GradientBoostingClassifier(random_state=7)
        return logit, gbt

    start = min_train_rows
    for i in range(start, len(data), int(max(1, step_days))):
        train = data.iloc[:i]
        test_row = data.iloc[[i]]

        X_train = train.drop(columns=["y"])
        y_train = train["y"].astype(int)
        X_test = test_row.drop(columns=["y"])

        if len(np.unique(y_train)) < 2:
            continue

        logit, gbt = _make_models()
        try:
            logit.fit(X_train, y_train)
            gbt.fit(X_train, y_train)
            p = 0.5 * logit.predict_proba(X_test)[:, 1][0] + 0.5 * gbt.predict_proba(X_test)[:, 1][0]
            probs.iloc[i] = float(p)
        except Exception:
            continue

    return probs.dropna()

def simulate_strategy_from_probs(
    close: pd.Series,
    ret: pd.Series,
    probs: pd.Series,
    thr_long: float = 0.60,
    thr_cash: float = 0.45,
    fee_bps: float = 8.0
) -> pd.DataFrame:
    c = close.astype(float).copy()
    r = ret.astype(float).copy()
    p = probs.reindex(c.index).ffill()

    pos = pd.Series(index=c.index, dtype=float)
    cur = 0.0
    for i, dt in enumerate(c.index):
        pv = p.loc[dt] if dt in p.index else np.nan
        if np.isfinite(pv):
            if pv > thr_long:
                cur = 1.0
            elif pv < thr_cash:
                cur = 0.0
            else:
                cur = cur
        pos.iloc[i] = cur

    fee = (fee_bps / 10000.0)
    turns = pos.diff().abs().fillna(0.0)
    strat_ret = pos.shift(1).fillna(0.0) * r - turns * fee

    eq = (1 + strat_ret.fillna(0)).cumprod()
    bh = (1 + r.fillna(0)).cumprod()

    out = pd.DataFrame({
        "close": c,
        "prob": p,
        "position": pos,
        "asset_ret": r,
        "strategy_ret": strat_ret,
        "equity_strategy": eq,
        "equity_buyhold": bh
    })
    return out

def perf_stats(equity: pd.Series, daily_ret: pd.Series) -> dict:
    eq = equity.dropna().astype(float)
    r = daily_ret.dropna().astype(float)
    if len(eq) < 60 or len(r) < 60:
        return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan, "win_rate": np.nan}
    days = len(eq)
    cagr = float(eq.iloc[-1] ** (365.0 / max(1, days)) - 1)
    sh = sharpe_simple(r, periods_per_year=365)
    mdd = float(drawdown_series(eq).min())
    win = float((r > 0).mean())
    return {"cagr": cagr, "sharpe": sh, "max_dd": mdd, "win_rate": win}

def calibration_table(prob: pd.Series, y_true: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    tmp = pd.concat([prob.rename("p"), y_true.rename("y")], axis=1).dropna()
    if len(tmp) < 200:
        return pd.DataFrame()
    tmp["bin"] = pd.qcut(tmp["p"], q=n_bins, duplicates="drop")
    g = tmp.groupby("bin").agg(
        p_mean=("p", "mean"),
        y_rate=("y", "mean"),
        n=("y", "size")
    ).reset_index(drop=True)
    return g

def brier_score(prob: pd.Series, y_true: pd.Series) -> float:
    tmp = pd.concat([prob.rename("p"), y_true.rename("y")], axis=1).dropna()
    if len(tmp) < 200:
        return np.nan
    p = tmp["p"].clip(1e-6, 1-1e-6).astype(float)
    y = tmp["y"].astype(float)
    return float(np.mean((p - y) ** 2))

def lift_deciles(prob: pd.Series, fwd_ret: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    tmp = pd.concat([prob.rename("p"), fwd_ret.rename("fwd")], axis=1).dropna()
    if len(tmp) < 200:
        return pd.DataFrame()
    tmp["decile"] = pd.qcut(tmp["p"], q=n_bins, labels=False, duplicates="drop") + 1
    out = tmp.groupby("decile").agg(
        p_mean=("p","mean"),
        fwd_mean=("fwd","mean"),
        fwd_median=("fwd","median"),
        n=("fwd","size")
    ).reset_index()
    return out.sort_values("decile")

def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 7) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return (np.nan, np.nan)
    means = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=len(x), replace=True)
        means.append(np.mean(samp))
    lo = float(np.quantile(means, alpha/2))
    hi = float(np.quantile(means, 1 - alpha/2))
    return lo, hi

def mann_whitney_u_test(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import mannwhitneyu
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if len(a) < 30 or len(b) < 30:
            return np.nan
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan

def conditional_forward_returns(df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    close = df["close"].astype(float)
    rsi14 = rsi(close, 14)
    adx14 = adx(df, 14)
    bbp = bollinger_pctb(close)

    fwd = (close.shift(-horizon) / close - 1.0).astype(float)

    conditions = {
        "RSI<30 (sobreventa)": (rsi14 < 30),
        "RSI>70 (sobrecompra)": (rsi14 > 70),
        "ADX>25 (tendencia fuerte)": (adx14 > 25),
        "BB%B<0.1 (cerca banda baja)": (bbp < 0.1),
        "BB%B>0.9 (cerca banda alta)": (bbp > 0.9),
    }

    rows = []
    base = fwd.dropna().values
    base_mean = np.mean(base) if len(base) else np.nan

    for name, mask in conditions.items():
        a = fwd[mask].dropna().values
        b = fwd[~mask].dropna().values
        if len(a) < 60 or len(b) < 60:
            rows.append({
                "Condición": name,
                "N cond": int(len(a)),
                "Media fwd": np.nan,
                "IC 95% (lo)": np.nan,
                "IC 95% (hi)": np.nan,
                "Media no-cond": np.nan,
                "p (M-W)": np.nan,
                "Diff vs base": np.nan,
            })
            continue
        lo, hi = bootstrap_mean_ci(a, n_boot=1500, alpha=0.05, seed=7)
        pval = mann_whitney_u_test(a, b)
        rows.append({
            "Condición": name,
            "N cond": int(len(a)),
            "Media fwd": float(np.mean(a)),
            "IC 95% (lo)": lo,
            "IC 95% (hi)": hi,
            "Media no-cond": float(np.mean(b)),
            "p (M-W)": pval,
            "Diff vs base": float(np.mean(a) - base_mean) if np.isfinite(base_mean) else np.nan,
        })
    return pd.DataFrame(rows)

def explain_cfr():
    return (
        "Estas pruebas responden a la pregunta clave: **“cuando aparece una señal, qué pasa después, en promedio?”**\n\n"
        "- No miramos el indicador por estética; miramos **retorno futuro**.\n"
        "- Incluimos un **intervalo de confianza (bootstrap)**: si es muy ancho, la señal es poco estable.\n"
        "- Incluimos una **prueba estadística** (Mann-Whitney) que compara la distribución de retornos cuando la condición se cumple vs cuando no.\n\n"
        "Cómo leerlo sin finanzas:\n"
        "- Si ‘Media fwd’ es positiva y el IC 95% está bastante por encima de 0, suele ser una señal más interesante.\n"
        "- Si el p-valor es bajo (por ejemplo < 0.05) sugiere que la diferencia no parece casual.\n"
        "- Si hay pocos casos (N cond pequeño), no te fíes: puede ser coincidencia."
    )


# =========================
# PPEI (Protocol-to-Price Elasticity Index) usando histórico guardado
# =========================
def compute_ppei_from_history(hist: pd.DataFrame, horizon: int = 30) -> dict:
    if hist is None or hist.empty:
        return {"ppei_coef": np.nan, "ppei_today": np.nan, "ppei_r2": np.nan, "ppei_n": 0, "ppei_status": "No history"}

    h = hist.copy()
    if "as_of_date" not in h.columns:
        return {"ppei_coef": np.nan, "ppei_today": np.nan, "ppei_r2": np.nan, "ppei_n": 0, "ppei_status": "Missing as_of_date"}
    h["as_of_date"] = pd.to_datetime(h["as_of_date"], errors="coerce")
    h = h.dropna(subset=["as_of_date"]).sort_values("as_of_date")

    candidates = []
    for col in ["fund_totalTokensStaked", "fund_activeIndexers", "fund_fund_stake_ratio", "fund_totalDelegatedTokens", "fund_totalTokensAllocated"]:
        if col in h.columns:
            candidates.append(col)

    for col in ["totalTokensStaked", "activeIndexers", "totalDelegatedTokens", "totalTokensAllocated"]:
        if col in h.columns and col not in candidates:
            candidates.append(col)

    if len(candidates) < 2:
        return {"ppei_coef": np.nan, "ppei_today": np.nan, "ppei_r2": np.nan, "ppei_n": 0, "ppei_status": "Not enough protocol fields in history"}

    if "close" in h.columns:
        close = pd.to_numeric(h["close"], errors="coerce")
        fwd_ret = close.shift(-horizon) / close - 1.0
    else:
        pcol = "p_up_30d" if "p_up_30d" in h.columns else None
        if pcol is None:
            return {"ppei_coef": np.nan, "ppei_today": np.nan, "ppei_r2": np.nan, "ppei_n": 0, "ppei_status": "No close and no p_up_30d to proxy"}
        fwd_ret = pd.to_numeric(h[pcol], errors="coerce") - 0.5

    feats = pd.DataFrame(index=h.index)
    for col in candidates:
        x = pd.to_numeric(h[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        feats[col] = np.log(x).diff()

    proto = feats.apply(lambda s: robust_z(s), axis=0).mean(axis=1)
    proto = proto.replace([np.inf, -np.inf], np.nan)

    dfm = pd.concat([proto.rename("proto"), fwd_ret.rename("y")], axis=1).dropna()
    if len(dfm) < 120:
        return {"ppei_coef": np.nan, "ppei_today": np.nan, "ppei_r2": np.nan, "ppei_n": int(len(dfm)), "ppei_status": "Too few rows"}

    X = sm.add_constant(dfm["proto"].astype(float))
    y = dfm["y"].astype(float)
    try:
        model = sm.OLS(y, X).fit(cov_type="HC3")
        coef = float(model.params["proto"])
        r2 = float(model.rsquared)
        proto_today = float(dfm["proto"].iloc[-1])
        ppei_today = float(coef * proto_today)
        return {"ppei_coef": coef, "ppei_today": ppei_today, "ppei_r2": r2, "ppei_n": int(len(dfm)), "ppei_status": "OK"}
    except Exception as e:
        return {"ppei_coef": np.nan, "ppei_today": np.nan, "ppei_r2": np.nan, "ppei_n": int(len(dfm)), "ppei_status": f"Fail: {e}"}

def explain_ppei():
    return (
        "**PPEI (Protocol-to-Price Elasticity Index)** intenta responder: *“cuando el protocolo mejora, ¿eso se traduce en mejor comportamiento?”*\n\n"
        "Cómo lo calculamos aquí:\n"
        "- Miramos cambios en métricas del protocolo (stake/indexers/delegación/allocations… según lo disponible).\n"
        "- Los convertimos en un ‘pulso’ normalizado (proto).\n"
        "- Estimamos si ese pulso se asocia con retorno futuro (o una proxy si no guardas ‘close’).\n\n"
        "Cómo interpretarlo:\n"
        "- **Coeficiente PPEI** alto y estable: mejoras del protocolo suelen ir acompañadas de mejores resultados.\n"
        "- **Coeficiente ~0**: el mercado no está ‘pagando’ esas mejoras (o faltan datos/hay ruido).\n"
        "- **PPEI hoy** combina: (elasticidad) × (pulso reciente). Si es positivo y grande, sería un “viento a favor”."
    )


# =========================
# SCORE
# =========================
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

    rsi14 = metrics.get("rsi_14", np.nan)
    if rsi14 == rsi14:
        if rsi14 < 30:
            s += 4
        elif rsi14 > 70:
            s -= 4

    adx14 = metrics.get("adx_14", np.nan)
    if adx14 == adx14:
        s += 3 if adx14 > 25 else 0

    news = metrics.get("news_score_0_100", np.nan)
    if news == news:
        if news >= 65:
            s += 6
        elif news >= 55:
            s += 3
        elif news <= 35:
            s -= 6
        elif news <= 45:
            s -= 3

    reg = metrics.get("hmm_regime", None)
    pr = metrics.get("hmm_p_regime", np.nan)
    if isinstance(reg, str):
        if "Uptrend" in reg:
            s += 8 if (pr == pr and pr > 0.6) else 5
        elif "Dump" in reg:
            s -= 8

    p30 = probs.get(30, np.nan)
    p90 = probs.get(90, np.nan)
    if p30 == p30:
        s += 8 if p30 > 0.6 else (-8 if p30 < 0.45 else 0)
    if p90 == p90:
        s += 6 if p90 > 0.6 else (-6 if p90 < 0.45 else 0)

    return float(np.clip(s, 0, 100))


# =========================
# CSV IO
# =========================
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
    st.caption("Regímenes · Probabilidades · Backtest · Risk · Fundamentals · News")

    preferred_exchange = st.selectbox(
        "Exchange preferido (fallback auto)",
        ["binance", "kraken", "coinbase", "bitstamp"],
        index=["binance","kraken","coinbase","bitstamp"].index(settings.get("preferred_exchange","binance"))
    )
    symbol = st.text_input("Símbolo", value=settings.get("symbol","GRT/USDT"))
    benchmark = st.text_input("Benchmark", value=settings.get("benchmark","BTC/USDT"))
    days = st.slider("Histórico (días)", 250, 2000, int(settings.get("days", 900)), step=50)

    st.divider()
    st.markdown("### 📰 Noticias (RSS)")
    news_cfg = settings.get("news", DEFAULT_SETTINGS["news"])
    news_enable = st.toggle("Activar noticias", value=bool(news_cfg.get("enable", True)))
    news_lookback = st.slider("Ventana titulares (días)", 3, 30, int(news_cfg.get("lookback_days", 14)))
    st.caption("Score 0–100 desde titulares dentro de la ventana seleccionada.")

    st.divider()
    st.markdown("### 🧪 Backtest Walk-Forward")
    bt_cfg = settings.get("backtest", DEFAULT_SETTINGS["backtest"])
    bt_h = st.selectbox("Horizonte del backtest (días)", [7, 30, 90], index=[7,30,90].index(int(bt_cfg.get("horizon_days",30))))
    bt_step = st.slider("Paso (días) — más alto = más rápido", 1, 14, int(bt_cfg.get("step_days", 7)))
    bt_mintrain = st.slider("Mínimo datos para entrenar", 250, 700, int(bt_cfg.get("min_train_rows", 320)), step=10)
    bt_thr_long = st.slider("Umbral entrar (prob)", 0.50, 0.80, float(bt_cfg.get("thr_long", 0.60)))
    bt_thr_cash = st.slider("Umbral salir (prob)", 0.20, 0.55, float(bt_cfg.get("thr_cash", 0.45)))
    bt_fee = st.slider("Coste por cambio de posición (bps)", 0, 30, int(bt_cfg.get("fee_bps", 8)))

    st.divider()
    st.markdown("### 🔌 The Graph Gateway (opcional)")
    st.caption("Si no tienes key, usamos CoinGecko automáticamente.")
    api_keys = settings.get("api_keys", {})
    thegraph_key = st.text_input("THE_GRAPH_API_KEY", value=api_keys.get("thegraph_gateway",""), type="password")

    st.divider()
    auto_save = st.toggle("Guardar resultado al actualizar", value=True)
    run_update = st.button("🔄 Actualizar (calcular todo)", type="primary")

    if st.button("💾 Guardar ajustes"):
        settings["preferred_exchange"] = preferred_exchange
        settings["symbol"] = symbol.strip()
        settings["benchmark"] = benchmark.strip()
        settings["days"] = int(days)
        settings.setdefault("api_keys", {})
        settings["api_keys"]["thegraph_gateway"] = thegraph_key.strip()
        settings["news"] = {"enable": bool(news_enable), "lookback_days": int(news_lookback), "rss_timeout": int(news_cfg.get("rss_timeout", 15))}
        settings["backtest"] = {
            "horizon_days": int(bt_h),
            "step_days": int(bt_step),
            "min_train_rows": int(bt_mintrain),
            "thr_long": float(bt_thr_long),
            "thr_cash": float(bt_thr_cash),
            "fee_bps": int(bt_fee),
        }
        save_settings(settings)
        st.success("Ajustes guardados.")


# =========================
# HERO
# =========================
st.markdown(
    """
    <div class="epic-hero">
      <div class="epic-title">🟣 GRT QuantLab <span class="badge">Market · Modelos · Backtest · Señales · Risk · Fundamentals · News</span></div>
      <div class="epic-sub">
        Arriba: Score + Probabilidades + News. Abajo: mercado, pruebas estadísticas, backtest walk-forward y explicaciones claras.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")


# =========================
# RUN
# =========================
metrics: Dict[str, object] = {}
probs = {7: np.nan, 30: np.nan, 90: np.nan}
report = {}
diag = {}
df = pd.DataFrame()
dfb = pd.DataFrame()
fund = pd.DataFrame()
fund_last = None
fund_source = "N/A"
score = np.nan

news_score = np.nan
news_df = pd.DataFrame()
news_daily = pd.DataFrame()

progress = st.empty()
status = st.empty()

if run_update:
    try:
        bar = progress.progress(0, text="Preparando…")
        status.info("Inicializando…")

        tf = settings.get("timeframe","1d")

        bar.progress(10, text="Descargando OHLCV…")
        status.info("Descargando datos de mercado…")
        df = compute_returns(fetch_ohlcv(preferred_exchange, symbol, tf, limit=int(days)))
        dfb = compute_returns(fetch_ohlcv(preferred_exchange, benchmark, tf, limit=int(days)))

        used_ex = df.attrs.get("exchange_used", preferred_exchange)
        used_sym = df.attrs.get("symbol_used", symbol)
        used_bench = dfb.attrs.get("symbol_used", benchmark)

        bar.progress(25, text="Cargando fundamentals…")
        status.info("Cargando fundamentals…")
        tgk = settings.get("api_keys", {}).get("thegraph_gateway","").strip()
        if tgk:
            try:
                fund = fetch_grt_network_fundamentals_gateway(tgk)
                fund_source = "Graph Gateway"
            except Exception as fe:
                fund = fetch_grt_fundamentals_coingecko()
                fund_source = f"CoinGecko (fallback: {fe})"
        else:
            fund = fetch_grt_fundamentals_coingecko()
            fund_source = "CoinGecko"

        fund_last = fund.iloc[-1] if (fund is not None and not fund.empty) else None

        bar.progress(40, text="Noticias (RSS)…")
        status.info("Leyendo titulares y calculando score…")
        news_cfg = settings.get("news", DEFAULT_SETTINGS["news"])
        try:
            news_score, news_df, news_daily = build_news_panel(
                enable=bool(news_cfg.get("enable", True)),
                lookback_days=int(news_cfg.get("lookback_days", 14)),
                timeout=int(news_cfg.get("rss_timeout", 15)),
            )
        except Exception:
            news_score, news_df, news_daily = np.nan, pd.DataFrame(), pd.DataFrame()

        bar.progress(55, text="Calculando métricas…")
        status.info("Calculando indicadores y riesgo…")
        metrics.update(trend_regression(df, 90))
        metrics.update(stationarity_tests(df))
        metrics.update(momentum_metrics(df))
        metrics.update(volume_signal(df))
        metrics.update(garch_volatility(df))
        metrics.update(structural_breaks(df))
        metrics.update(tail_risk_metrics(df))
        metrics.update(extra_market_stats(df, dfb))
        metrics.update(bench_relationship_tests(df, dfb))
        metrics["news_score_0_100"] = float(news_score) if np.isfinite(news_score) else np.nan

        if fund_last is not None and isinstance(fund_last, pd.Series) and not fund_last.empty:
            for k, v in fund_last.to_dict().items():
                metrics[f"fund_{k}"] = safe_float(v)

            if ("fund_totalTokensStaked" in metrics) and ("fund_totalSupply" in metrics):
                metrics["fund_stake_ratio"] = safe_float(metrics["fund_totalTokensStaked"]) / (safe_float(metrics["fund_totalSupply"]) + 1e-12)
            if ("fund_cg_marketcap_usd" in metrics) and ("fund_cg_volume_24h_usd" in metrics):
                metrics["fund_mcap_to_vol"] = safe_float(metrics["fund_cg_marketcap_usd"]) / (safe_float(metrics["fund_cg_volume_24h_usd"]) + 1e-12)

        bar.progress(65, text="HMM (regímenes)…")
        status.info("Estimando regímenes…")
        metrics.update(hmm_regimes_robust(df))

        bar.progress(78, text="Entrenando modelos…")
        status.info("Entrenando ensemble + explicabilidad…")
        feat = build_feature_frame(df, dfb, fund_last, news_score_0_100=news_score)
        probs, report, diag = fit_ensemble_probabilities(feat, horizons=(7,30,90), min_rows=250)

        bar.progress(90, text="Calculando score…")
        status.info("Cerrando cálculo…")
        score = grt_score(metrics, probs)

        if auto_save:
            bar.progress(95, text="Guardando…")
            row = {
                "as_of_date": str(df.index.max()),
                "exchange": used_ex,
                "symbol": used_sym,
                "benchmark": used_bench,
                "updated_at_utc": now_utc_iso(),
                "fund_source": fund_source,
                "score_0_100": float(score),
                "news_score_0_100": safe_float(news_score),
                "p_up_7d": safe_float(probs.get(7)),
                "p_up_30d": safe_float(probs.get(30)),
                "p_up_90d": safe_float(probs.get(90)),
                **{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in metrics.items()},
            }
            save_row_to_csv(row)

        bar.progress(100, text="Listo ✅")
        status.success(f"✅ Actualizado ({used_ex} · {used_sym}) — Score: {score:.1f}/100")
        st.caption(f"Benchmark: {dfb.attrs.get('exchange_used', used_ex)} · {used_bench} · Fundamentals: {fund_source}")

        hmm_status = metrics.get("hmm_status", "")
        if isinstance(hmm_status, str) and hmm_status.startswith("HMM failed"):
            st.warning(hmm_status)

    except Exception as e:
        status.error(f"Error al actualizar: {e}")

hist = load_results()
if (not run_update) and (not hist.empty):
    last = hist.sort_values("as_of_date").iloc[-1].to_dict()
    score = safe_float(last.get("score_0_100"))
    news_score = safe_float(last.get("news_score_0_100"))
    probs = {7: safe_float(last.get("p_up_7d")), 30: safe_float(last.get("p_up_30d")), 90: safe_float(last.get("p_up_90d"))}


# =========================
# KPIs ARRIBA
# =========================
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>GRT Score</div>
          <div class='value'>{(score if score==score else np.nan):.1f}/100</div>
          <div class='hint'>Confluencia</div>
          <div class='hint'>{explain_score(score)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    p7 = probs.get(7, np.nan)
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>P(↑ 7 días)</div>
          <div class='value'>{_fmt_pct(p7)}</div>
          <div class='hint'>Ensemble</div>
          <div class='hint'>{explain_prob(p7, 7)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    p30 = probs.get(30, np.nan)
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>P(↑ 30 días)</div>
          <div class='value'>{_fmt_pct(p30)}</div>
          <div class='hint'>Ensemble</div>
          <div class='hint'>{explain_prob(p30, 30)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    p90 = probs.get(90, np.nan)
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>P(↑ 90 días)</div>
          <div class='value'>{_fmt_pct(p90)}</div>
          <div class='hint'>Ensemble</div>
          <div class='hint'>{explain_prob(p90, 90)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c5:
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>News Score</div>
          <div class='value'>{_fmt_num(news_score, 1) if np.isfinite(news_score) else "—"}</div>
          <div class='hint'>0–100 (ventana)</div>
          <div class='hint'>{explain_news_score(news_score)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

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


# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Market",
    "🧠 Modelos",
    "🧪 Señales (pruebas)",
    "🧭 Backtest (walk-forward)",
    "📰 Noticias",
    "🧾 Histórico / Export",
])

with tab1:
    st.subheader("Market")
    if df.empty:
        st.info("Pulsa **Actualizar (calcular todo)**.")
    else:
        cd = df.reset_index()
        fig_c = go.Figure(data=[go.Candlestick(
            x=cd["date"], open=cd["open"], high=cd["high"], low=cd["low"], close=cd["close"]
        )])
        fig_c.update_layout(height=420, margin=dict(l=20,r=20,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c, use_container_width=True)

        fig_v = go.Figure()
        fig_v.add_trace(go.Bar(x=cd["date"], y=cd["volume"]))
        fig_v.update_layout(height=220, margin=dict(l=20,r=20,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_v, use_container_width=True)

        if len(df) > 60:
            rsi_s = rsi(df["close"].astype(float), 14)
            m_line, m_sig, m_hist = macd(df["close"].astype(float))
            mini = pd.DataFrame({
                "date": df.index.astype(str),
                "RSI14": rsi_s.values,
                "MACD_hist": m_hist.values
            }).tail(180)

            st.write("")
            st.markdown("#### Indicadores rápidos")
            fig_rsi = px.line(mini, x="date", y="RSI14", title="RSI(14) (últimos 180 días)")
            fig_rsi.update_layout(height=260, margin=dict(l=20,r=20,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rsi, use_container_width=True)

            fig_mh = px.bar(mini, x="date", y="MACD_hist", title="MACD Hist (últimos 180 días)")
            fig_mh.update_layout(height=260, margin=dict(l=20,r=20,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_mh, use_container_width=True)

with tab2:
    st.subheader("Modelos · Probabilidades + Explicabilidad")

    if not report or df.empty:
        st.info("Pulsa **Actualizar (calcular todo)**.")
    else:
        rows = []
        for h in [7, 30, 90]:
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

        st.write("")
        st.markdown("### Interpretación rápida por horizonte")

        for h in [7, 30, 90]:
            rep = report.get(h, {})
            p = probs.get(h, np.nan)
            auc = rep.get("auc", np.nan)
            n_train = rep.get("n_train", 0)
            n_test = rep.get("n_test", 0)
            rows_clean = diag.get(h, {}).get("rows_after_clean", 0)

            text = model_summary_text(p, auc, n_train, n_test, rows_clean, h)

            st.markdown(
                f"""
                <div class='kpi'>
                  <div class='label'>Horizonte {h} días</div>
                  <div class='value'>
                    P(↑) {_fmt_pct(p)} · AUC {(auc if np.isfinite(auc) else np.nan):.3f}
                  </div>
                  <div class='hint'>{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")
        st.markdown("### Factores más influyentes")

        h_sel = st.selectbox("Ver factores top para horizonte", [7, 30, 90], index=1)

        rep_sel = report.get(h_sel, {})
        p_sel = probs.get(h_sel, np.nan)
        auc_sel = rep_sel.get("auc", np.nan)
        rows_sel = diag.get(h_sel, {}).get("rows_after_clean", 0)

        st.info(
            model_summary_text(
                p_sel,
                auc_sel,
                rep_sel.get("n_train", 0),
                rep_sel.get("n_test", 0),
                rows_sel,
                h_sel
            )
        )

        top_feats = rep_sel.get("top_features", [])
        if not top_feats:
            st.warning("No se pudo calcular importance (AUC inválido o pocos datos).")
        else:
            imp_df = pd.DataFrame(top_feats, columns=["feature", "importance"])
            fig_imp = px.bar(
                imp_df,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Top factores (perm. importance) — {h_sel}d"
            )
            fig_imp.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_imp, use_container_width=True)

        st.caption(f"HMM status: {metrics.get('hmm_status','N/A')}")

        with st.expander("📌 ¿Qué significa esto en cristiano? (explicación larga)"):
            st.write(
                "1) **P(↑)** es la probabilidad que el modelo estima para que el precio esté más alto dentro del horizonte.\n\n"
                "2) **AUC** mide si el modelo separa ‘subida’ vs ‘no subida’ en datos de test. Si AUC≈0.50, es casi azar.\n\n"
                "3) **Top factores** te dice qué variables están empujando más la predicción (aprox). "
                "Esto no es una ‘verdad’, pero ayuda a entender si el modelo se apoya en volumen, tendencia, volatilidad, etc.\n\n"
                "Consejo: si P(↑) es alta pero AUC es baja, el modelo puede estar **sobreconfiado**."
            )

with tab3:
    st.subheader("Señales con pruebas estadísticas (CFR + relación BTC + riesgo extremo)")

    if df.empty or not metrics:
        st.info("Pulsa **Actualizar (calcular todo)**.")
    else:
        st.markdown("### 1) CFR: cuando pasa X… ¿qué pasa después?")
        cfr_df = conditional_forward_returns(df, horizon=30)
        st.dataframe(cfr_df, use_container_width=True)
        st.caption(explain_cfr())

        st.divider()
        st.markdown("### 2) Relación GRT vs Benchmark (BTC) — ¿copia o va por libre?")
        cA, cB = st.columns(2)
        with cA:
            coint_p = metrics.get("coint_pvalue", np.nan)
            st.markdown(
                f"<div class='kpi'><div class='label'>Cointegración (p-valor)</div><div class='value'>{_fmt_num(coint_p,3)}</div>"
                f"<div class='hint'>Si p &lt; 0.05, sugiere relación de largo plazo (tienden a ‘reunirse’ tras separarse). "
                f"Si p alto, pueden ir más independientes.</div></div>",
                unsafe_allow_html=True
            )
        with cB:
            gr_p = metrics.get("granger_min_pvalue", np.nan)
            st.markdown(
                f"<div class='kpi'><div class='label'>Granger (bench→GRT) min p</div><div class='value'>{_fmt_num(gr_p,3)}</div>"
                f"<div class='hint'>Si p &lt; 0.05, los movimientos del benchmark suelen *preceder* a GRT (dependencia temporal). "
                f"Si p alto, menos evidencia de que BTC ‘mande’.</div></div>",
                unsafe_allow_html=True
            )

        with st.expander("📌 Interpretación sencilla (larga)"):
            st.write(
                "Estas pruebas no te dicen ‘sube o baja’, te dicen **de qué depende**.\n\n"
                "- **Cointegración**: si existe, cuando GRT se separa mucho de BTC, a veces tiende a volver. "
                "Esto se parece a ‘ir atados por una goma’ a largo plazo.\n\n"
                "- **Granger**: si sale significativo, BTC suele moverse primero y GRT le sigue. "
                "Eso te dice que parte del riesgo de GRT es realmente *riesgo BTC*."
            )

        st.divider()
        st.markdown("### 3) Riesgo extremo (no solo VaR/CVaR)")
        dd = metrics.get("max_drawdown", np.nan)
        dd_max_dur = metrics.get("dd_max_duration", np.nan)
        dd_avg_dur = metrics.get("dd_avg_duration", np.nan)
        jump_rate = metrics.get("jump_rate", np.nan)
        jumps90 = metrics.get("jumps_last90", np.nan)
        evt_var = metrics.get("evt_var_99", np.nan)
        evt_es = metrics.get("evt_es_99", np.nan)
        evt_status = metrics.get("evt_status", "N/A")

        c1_, c2_, c3_, c4_ = st.columns(4)
        with c1_:
            st.markdown(f"<div class='kpi'><div class='label'>Max Drawdown</div><div class='value'>{_fmt_pct(dd)}</div><div class='hint'>La peor caída desde máximos. Te da el ‘dolor máximo’ histórico reciente.</div></div>", unsafe_allow_html=True)
        with c2_:
            st.markdown(f"<div class='kpi'><div class='label'>Duración DD (max)</div><div class='value'>{_fmt_num(dd_max_dur,0)}</div><div class='hint'>Cuántos días seguidos puede estar por debajo del máximo. Más = más tiempo ‘atrapado’.</div></div>", unsafe_allow_html=True)
        with c3_:
            st.markdown(f"<div class='kpi'><div class='label'>Jump rate</div><div class='value'>{_fmt_pct(jump_rate)}</div><div class='hint'>% de días con ‘saltos’ muy anómalos. Más alto = más impredecible para modelos.</div></div>", unsafe_allow_html=True)
        with c4_:
            st.markdown(f"<div class='kpi'><div class='label'>Jumps (últimos 90d)</div><div class='value'>{_fmt_num(jumps90,0)}</div><div class='hint'>Cuántos días recientes tuvieron movimientos extremos. Si sube, aumenta riesgo de latigazos.</div></div>", unsafe_allow_html=True)

        st.write("")
        c5_, c6_ = st.columns(2)
        with c5_:
            st.markdown(
                f"<div class='kpi'><div class='label'>EVT VaR 99% (diario)</div><div class='value'>{_fmt_pct(evt_var)}</div>"
                f"<div class='hint'>Estimación de pérdida rara (1 de cada 100 días malos) usando cola extrema. "
                f"Status: {evt_status}</div></div>",
                unsafe_allow_html=True
            )
            
                with c6_:
            st.markdown(
                f"""
                <div class='kpi'>
                  <div class='label'>EVT ES 99% (diario)</div>
                  <div class='value'>{_fmt_pct(evt_es)}</div>
                  <div class='hint'>
                    Si ocurre un día extremo, esta es la pérdida media esperada (peor que el VaR).
                    <br/>Status: {evt_status}
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.expander("📌 Nota (larga) sobre EVT"):
            st.write(
                "EVT (Extreme Value Theory) intenta modelar **la cola** (los días realmente raros).\n\n"
                "- **VaR 99%**: umbral de pérdida que esperas superar ~1% de los días.\n"
                "- **ES 99%**: si estás en ese 1% peor, cuál es la pérdida media.\n\n"
                "Si el estado dice 'Not enough data' o 'Too few exceedances', no hay suficiente histórico de cola."
            )

        st.divider()
        st.markdown("### 4) PPEI (protocolo → precio) desde tu histórico guardado")

        ppei = compute_ppei_from_history(hist, horizon=30) if (hist is not None and not hist.empty) else {
            "ppei_coef": np.nan, "ppei_today": np.nan, "ppei_r2": np.nan, "ppei_n": 0, "ppei_status": "No history"
        }

        cp1, cp2, cp3, cp4 = st.columns(4)
        with cp1:
            st.markdown(
                f"""
                <div class='kpi'>
                  <div class='label'>PPEI coef</div>
                  <div class='value'>{_fmt_num(ppei.get('ppei_coef', np.nan), 4)}</div>
                  <div class='hint'>Elasticidad (proto→fwd). Status: {ppei.get('ppei_status','')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with cp2:
            st.markdown(
                f"""
                <div class='kpi'>
                  <div class='label'>PPEI hoy</div>
                  <div class='value'>{_fmt_num(ppei.get('ppei_today', np.nan), 4)}</div>
                  <div class='hint'>(coef × pulso proto reciente)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with cp3:
            st.markdown(
                f"""
                <div class='kpi'>
                  <div class='label'>R²</div>
                  <div class='value'>{_fmt_num(ppei.get('ppei_r2', np.nan), 3)}</div>
                  <div class='hint'>Explicación (ojo: ruido alto en cripto)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with cp4:
            st.markdown(
                f"""
                <div class='kpi'>
                  <div class='label'>N</div>
                  <div class='value'>{_fmt_num(ppei.get('ppei_n', np.nan), 0)}</div>
                  <div class='hint'>Filas usadas para estimar</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.expander("📌 ¿Qué es PPEI? (explicación larga)"):
            st.write(explain_ppei())


with tab4:
    st.subheader("Backtest walk-forward (probabilidades → señal → equity)")

    if df.empty:
        st.info("Pulsa **Actualizar (calcular todo)**.")
    else:
        feat_bt = build_feature_frame(df, dfb, fund_last, news_score_0_100=news_score)

        st.write("Calculando serie de probabilidades walk-forward… (puede tardar)")
        prob_series = walk_forward_prob_series(
            feat_bt,
            horizon=int(bt_h),
            step_days=int(bt_step),
            min_train_rows=int(bt_mintrain)
        )

        if prob_series.empty:
            st.warning("No hay suficientes datos para el walk-forward con tus parámetros.")
        else:
            sim = simulate_strategy_from_probs(
                close=df["close"],
                ret=df["ret"],
                probs=prob_series,
                thr_long=float(bt_thr_long),
                thr_cash=float(bt_thr_cash),
                fee_bps=float(bt_fee)
            )

            st.markdown("### Equity curve")
            eq_df = sim.reset_index().rename(columns={"index": "date"})
            fig_eq = px.line(eq_df, x="date", y=["equity_strategy", "equity_buyhold"], title="Estrategia vs Buy&Hold")
            fig_eq.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_eq, use_container_width=True)

            st.markdown("### Probabilidades y posición")
            fig_p = px.line(eq_df, x="date", y="prob", title=f"Probabilidad walk-forward (h={bt_h}d)")
            fig_p.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_p, use_container_width=True)

            fig_pos = px.area(eq_df, x="date", y="position", title="Posición (0=cash, 1=long)")
            fig_pos.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pos, use_container_width=True)

            st.markdown("### Métricas de performance")
            ps = perf_stats(sim["equity_strategy"], sim["strategy_ret"])
            pb = perf_stats(sim["equity_buyhold"], sim["asset_ret"])

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(
                    f"""
                    <div class='kpi'>
                      <div class='label'>Estrategia</div>
                      <div class='value'>CAGR {_fmt_pct(ps.get('cagr', np.nan))}</div>
                      <div class='hint'>Sharpe {_fmt_num(ps.get('sharpe', np.nan),2)} · MaxDD {_fmt_pct(ps.get('max_dd', np.nan))} · WinRate {_fmt_pct(ps.get('win_rate', np.nan))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with r2:
                st.markdown(
                    f"""
                    <div class='kpi'>
                      <div class='label'>Buy & Hold</div>
                      <div class='value'>CAGR {_fmt_pct(pb.get('cagr', np.nan))}</div>
                      <div class='hint'>Sharpe {_fmt_num(pb.get('sharpe', np.nan),2)} · MaxDD {_fmt_pct(pb.get('max_dd', np.nan))} · WinRate {_fmt_pct(pb.get('win_rate', np.nan))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.divider()
            st.markdown("### Calibración (bins) + Brier score")

            # y_true para el mismo horizonte, alineado
            _, y_all = _prep_xy_for_horizon(feat_bt, int(bt_h))
            y_true = y_all.reindex(prob_series.index)

            cal = calibration_table(prob_series, y_true, n_bins=10)
            bs = brier_score(prob_series, y_true)

            cA, cB = st.columns([2, 1])
            with cA:
                if cal.empty:
                    st.warning("No hay suficientes datos para calibración.")
                else:
                    st.dataframe(cal, use_container_width=True)
                    cal_plot = cal.copy()
                    cal_plot["bin_id"] = np.arange(1, len(cal_plot) + 1)
                    fig_cal = px.line(cal_plot, x="bin_id", y=["p_mean", "y_rate"], title="Calibración: prob media vs tasa real")
                    fig_cal.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_cal, use_container_width=True)
            with cB:
                st.markdown(
                    f"""
                    <div class='kpi'>
                      <div class='label'>Brier score</div>
                      <div class='value'>{_fmt_num(bs, 4)}</div>
                      <div class='hint'>Más bajo = mejor calibración</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.divider()
            st.markdown("### Lift por deciles (¿las probs altas dan mejores returns?)")
            fwd_ret = (df["close"].shift(-int(bt_h)) / df["close"] - 1.0).astype(float)
            lift = lift_deciles(prob_series, fwd_ret.reindex(prob_series.index), n_bins=10)
            if lift.empty:
                st.warning("No hay suficientes datos para lift.")
            else:
                st.dataframe(lift, use_container_width=True)
                fig_l = px.bar(lift, x="decile", y="fwd_mean", title="Forward return medio por decil de prob")
                fig_l.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_l, use_container_width=True)

            with st.expander("📌 Cómo leer el backtest (sin humo)"):
                st.write(
                    "- Walk-forward significa que cada predicción usa SOLO el pasado, como en vida real.\n"
                    "- La estrategia entra/sale por umbrales. Si subes el umbral, hay menos trades pero más selectivos.\n"
                    "- Mira MaxDD y duración de DD: en cripto importa tanto como el retorno.\n"
                    "- Si Lift no mejora en deciles altos, el modelo probablemente no añade ventaja."
                )


with tab5:
    st.subheader("Noticias (RSS) · Titulares y serie diaria")

    if news_df is None or news_df.empty:
        st.info("No hay noticias en la ventana (o RSS desactivado / falló).")
    else:
        st.markdown("### Score actual")
        st.markdown(
            f"""
            <div class='kpi'>
              <div class='label'>News Score (0–100)</div>
              <div class='value'>{_fmt_num(news_score, 1) if np.isfinite(news_score) else "—"}</div>
              <div class='hint'>{explain_news_score(news_score)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        st.markdown("### Serie diaria (media)")
        if news_daily is not None and not news_daily.empty:
            nd = news_daily.reset_index().rename(columns={"index": "date"})
            nd["date"] = nd["date"].astype(str)
            fig_nd = px.line(nd, x="date", y="score_0_100", title="News score diario (media)")
            fig_nd.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_nd, use_container_width=True)

        st.write("")
        st.markdown("### Titulares (filtrables)")
        q = st.text_input("Buscar en titulares", value="")
        df_show = news_df.copy()
        if q.strip():
            qq = _clean_text(q.strip())
            df_show = df_show[df_show["title"].apply(lambda s: qq in _clean_text(str(s)))]

        df_show = df_show.sort_values("date", ascending=False)
        st.dataframe(df_show[["date", "title", "sent_0_100", "link"]], use_container_width=True)


with tab6:
    st.subheader("Histórico / Export")

    if hist is None or hist.empty:
        st.info("Aún no hay histórico. Activa “Guardar resultado al actualizar” y pulsa **Actualizar**.")
    else:
        h = hist.copy()
        # Orden robusto
        h["as_of_date"] = pd.to_datetime(h["as_of_date"], errors="coerce")
        h = h.dropna(subset=["as_of_date"]).sort_values("as_of_date", ascending=False)

        st.markdown("### Últimos resultados guardados")
        st.dataframe(h.head(200), use_container_width=True)

        st.write("")
        st.markdown("### Gráfico Score + Probabilidades")
        hh = h.sort_values("as_of_date")
        plot = pd.DataFrame({
            "date": hh["as_of_date"].dt.date.astype(str),
            "score": pd.to_numeric(hh.get("score_0_100", np.nan), errors="coerce"),
            "p7": pd.to_numeric(hh.get("p_up_7d", np.nan), errors="coerce"),
            "p30": pd.to_numeric(hh.get("p_up_30d", np.nan), errors="coerce"),
            "p90": pd.to_numeric(hh.get("p_up_90d", np.nan), errors="coerce"),
            "news": pd.to_numeric(hh.get("news_score_0_100", np.nan), errors="coerce"),
        })

        fig_hist = px.line(plot, x="date", y=["score", "news"], title="Score y News (histórico)")
        fig_hist.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_probs = px.line(plot, x="date", y=["p7", "p30", "p90"], title="Probabilidades (histórico)")
        fig_probs.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_probs, use_container_width=True)

        st.write("")
        st.markdown("### Exportar CSV")
        csv_bytes = h.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar histórico (CSV)",
            data=csv_bytes,
            file_name="grt_quantlab_history.csv",
            mime="text/csv"
        )

        st.caption("Tip: si cambias de máquina o repo, copia la carpeta /data para mantener el histórico.")


# =========================
# FOOTER
# =========================
st.write("")
st.caption("⚠️ Esto no es asesoramiento financiero. Es una app educativa/analítica: en cripto, el riesgo de pérdida es alto.")


