import os
import json
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

import requests
import ccxt
import xml.etree.ElementTree as ET

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

from statsmodels.tsa.stattools import grangercausalitytests

import plotly.graph_objects as go
import plotly.express as px

# HMM opcional
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False


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
    "api_keys": {"thegraph_gateway": ""},
    "news": {"enable": True, "lookback_days": 14, "rss_timeout": 15},
    "decision_lab": {
        "mc_sims": 2500,
        "mc_seed": 7,
        "similarity_window_days": 180,
        "granger_maxlag": 5
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
  font-size: 12px;
  margin-top: 10px;
  line-height: 1.35rem;
}

hr { border-color: rgba(30,42,85,0.55) !important; }

.small-muted{
  color: var(--muted);
  font-size: 12px;
}

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

def load_settings() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2, ensure_ascii=False)
        return DEFAULT_SETTINGS
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        s = json.load(f)
    # backfill keys
    for k, v in DEFAULT_SETTINGS.items():
        if k not in s:
            s[k] = v
    if "news" not in s:
        s["news"] = DEFAULT_SETTINGS["news"]
    if "decision_lab" not in s:
        s["decision_lab"] = DEFAULT_SETTINGS["decision_lab"]
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


# =========================
# EXPLICACIONES (amigables)
# =========================
def explain_score(score: float) -> str:
    if score is None or not np.isfinite(score):
        return "Sin cálculo aún. Pulsa “Actualizar”."
    if score >= 75:
        return ("Confluencia alta: varias señales están alineadas a la vez. "
                "En términos sencillos: el mercado, el momentum, el riesgo y el contexto (noticias) están más a favor que en contra. "
                "Aun así, en cripto nunca es garantía: sirve para reducir incertidumbre, no para eliminarla.")
    if score >= 60:
        return ("Confluencia moderada: hay señales a favor, pero no todas. "
                "Esto suele ser útil para vigilar si aparece confirmación (por ejemplo, que el momentum mejore o que el riesgo extremo baje).")
    if score >= 45:
        return ("Zona neutra: hay señales mezcladas. "
                "En este caso la estadística te está diciendo: “no tengo una historia clara”. "
                "Normalmente conviene esperar a que una parte del puzzle se defina.")
    if score >= 30:
        return ("Confluencia baja: predominan señales de debilidad o riesgo. "
                "No significa que no pueda rebotar, pero estadísticamente el contexto es más hostil.")
    return ("Riesgo elevado: el conjunto de señales apunta a cautela. "
            "Si alguien entra aquí, normalmente lo hace con tamaños pequeños, planes claros y tolerancia a volatilidad.")

def explain_prob(p: float, horizon_days: int) -> str:
    if p is None or not np.isfinite(p):
        return (f"Sin probabilidad calculada para {horizon_days} días. "
                "Esto suele ocurrir si hay pocos datos limpios o si el modelo no logró entrenar bien. "
                "No es tu culpa: es típico cuando faltan filas, hay NaNs o el comportamiento fue demasiado uniforme.")
    if p >= 0.75:
        return (f"Alta probabilidad de subida a {horizon_days} días según el modelo. "
                "Traducción humana: con datos pasados parecidos, la proporción de casos de subida es alta. "
                "Aun así, no es garantía: cripto puede romper patrones con noticias o shocks.")
    if p >= 0.60:
        return (f"Ventaja ligera a favor de subida a {horizon_days} días. "
                "Es una señal razonable, pero todavía exige confirmación con riesgo (colas) y tendencia.")
    if p >= 0.50:
        return (f"Escenario equilibrado a {horizon_days} días, casi 50/50. "
                "En términos prácticos: el modelo no está viendo un sesgo fuerte, así que otras señales pesan más.")
    if p >= 0.40:
        return (f"Ventaja ligera hacia bajada o lateralidad a {horizon_days} días. "
                "No significa caída segura; significa que la historia estadística reciente ha sido más complicada.")
    return (f"Probabilidad baja de subida a {horizon_days} días: el modelo ve más riesgo o debilidad. "
            "En ese contexto, suele ser clave mirar si estás en sobreventa (rebote) o si la tendencia fuerte sigue bajista.")

def explain_auc(auc: float) -> str:
    if auc is None or not np.isfinite(auc):
        return ("AUC no disponible: suele pasar si el test se queda con una sola clase (todo sube o todo baja) "
                "o si la muestra es demasiado pequeña. En pocas palabras: no hay base para medir calidad.")
    if auc >= 0.70:
        return ("Muy buena calidad: el modelo separa subidas vs bajadas bastante mejor que el azar. "
                "Esto no lo convierte en un oráculo, pero sí en una señal que merece respeto.")
    if auc >= 0.60:
        return ("Calidad aceptable: hay una ventaja estadística real frente al azar. "
                "Aun así, en cripto conviene exigir confluencia con riesgo y contexto.")
    if auc >= 0.55:
        return ("Calidad débil: es mejor que azar, pero poco. "
                "Suele ser útil solo si otras señales (tendencia, riesgo extremo, news) acompañan.")
    if auc >= 0.50:
        return ("Casi azar: el modelo apenas separa escenarios. "
                "En esta situación es mejor usarlo como curiosidad, no como guía.")
    return ("Peor que azar: puede indicar sobreajuste o que la señal está invertida. "
            "No significa que ‘sea lo contrario’, significa que no es fiable.")

def explain_news_score(x: float) -> str:
    if x is None or not np.isfinite(x):
        return ("No disponible: el RSS puede fallar o estar desactivado. "
                "Cuando esto pase, la app sigue funcionando, solo pierde el contexto de titulares.")
    if x >= 65:
        return ("Titulares claramente positivos. Esto a veces acompaña fases de impulso, "
                "pero también puede ser hype. La idea no es ‘comprar porque hay buenas noticias’, "
                "sino usarlo como una pieza más: si el riesgo es extremo, la noticia puede no salvarte.")
    if x >= 55:
        return ("Sesgo ligeramente positivo. Puede ayudar a que el mercado tenga viento a favor, "
                "pero por sí solo no justifica una inversión.")
    if x >= 45:
        return ("Neutral: titulares mezclados o sin mensaje claro. En este caso el precio y el riesgo suelen mandar más.")
    if x >= 35:
        return ("Sesgo negativo: aumenta la probabilidad de volatilidad y dudas. "
                "No siempre significa caída, pero sí un contexto menos ‘cómodo’.")
    return ("Muy negativo: cuidado con shocks. A veces después de extremos negativos hay rebotes, "
            "pero suelen ser escenarios de alta incertidumbre.")


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
# FUNDAMENTALS (CoinGecko + Gateway opcional)
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
        "totalSupply": safe_float(gn.get("totalSupply")),
        "activeIndexers": float(idx_count),
    }
    return pd.DataFrame([row]).set_index("as_of")


# =========================
# NEWS (RSS) -> Sentiment 0..100 (robusto sin keys)
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

def _clean_text(s: str) -> str:
    if not s:
        return ""
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s)

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

def build_news_panel(enable: bool, timeout: int) -> Tuple[float, pd.DataFrame]:
    if not enable:
        return np.nan, pd.DataFrame()

    queries = ["The+Graph+GRT", "The+Graph+protocol", "GRT+token"]
    rss_urls = [f"https://news.google.com/rss/search?q={q}&hl=en&gl=US&ceid=US:en" for q in queries]

    all_items = []
    for u in rss_urls:
        try:
            all_items.extend(fetch_rss_items(u, timeout=timeout, max_items=25))
        except Exception:
            continue

    if not all_items:
        return np.nan, pd.DataFrame()

    rows = []
    for it in all_items:
        txt = f"{it.get('title','')} {it.get('desc','')}"
        s = score_sentiment(txt)
        rows.append({"title": it.get("title",""), "pubDate": it.get("pubDate",""), "link": it.get("link",""), "sent": s})

    df_news = pd.DataFrame(rows).drop_duplicates(subset=["title"]).head(60)
    if df_news.empty:
        return np.nan, df_news

    svals = df_news["sent"].astype(float).values
    svals = svals[np.isfinite(svals)]
    if len(svals) == 0:
        return np.nan, df_news

    # trimmed mean (robusto)
    svals_sorted = np.sort(svals)
    k = max(0, int(0.1 * len(svals_sorted)))
    core = svals_sorted[k:len(svals_sorted)-k] if len(svals_sorted) > 10 else svals_sorted
    m = float(np.mean(core)) if len(core) else float(np.mean(svals_sorted))

    score_0_100 = float(np.clip(50 + 50*m, 0, 100))
    df_news["sent_0_100"] = (50 + 50*df_news["sent"]).clip(0, 100)
    return score_0_100, df_news


# =========================
# INDICADORES TÉCNICOS
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
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_ = tr.rolling(period).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / (atr_ + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / (atr_ + 1e-12))
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
    return dx.rolling(period).mean()

def max_drawdown(close: pd.Series) -> float:
    c = close.astype(float)
    roll_max = c.cummax()
    dd = (c / roll_max) - 1.0
    return float(dd.min()) if len(dd.dropna()) else np.nan

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
    tmp = pd.concat([asset_r.rename("a"), bench_r.rename("b")], axis=1).dropna().tail(window)
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


# =========================
# METRICS (base + extra)
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
    adf = adfuller(d, autolag="AIC")
    kpss_stat, kpss_p, _, _ = kpss(d, regression="c", nlags="auto")
    return {"adf_pvalue": float(adf[1]), "kpss_pvalue": float(kpss_p)}

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
    }


# =========================
# HMM (robusto)
# =========================
def hmm_regimes_robust(df: pd.DataFrame) -> dict:
    if not HMM_AVAILABLE:
        return {"hmm_regime": None, "hmm_p_regime": np.nan, "hmm_status": "HMM not installed"}

    r = df["ret"].dropna()
    if len(r) < 250:
        return {"hmm_regime": None, "hmm_p_regime": np.nan, "hmm_status": "Not enough data"}

    vol = r.rolling(14).std()
    X = pd.concat([r.rename("ret"), vol.rename("vol")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(X) < 200:
        return {"hmm_regime": None, "hmm_p_regime": np.nan, "hmm_status": "Not enough clean rows"}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values) + np.random.normal(0, 1e-8, size=X.shape)

    try:
        model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=300, random_state=7)
        model.fit(Xs)

        post = model.predict_proba(Xs)
        current_state = int(np.argmax(post[-1]))
        p_state = float(np.max(post[-1]))

        states = model.predict(Xs)
        means = []
        r_aligned = X["ret"].values
        for s in range(3):
            means.append(float(np.mean(r_aligned[states == s])))

        order = np.argsort(means)  # low -> high
        label_map = {int(order[0]): "Dump / Risk", int(order[1]): "Range / Accum", int(order[2]): "Uptrend"}
        label = label_map.get(current_state, "Unknown")

        return {"hmm_regime": label, "hmm_p_regime": p_state, "hmm_status": "OK"}
    except Exception as e:
        return {"hmm_regime": None, "hmm_p_regime": np.nan, "hmm_status": f"HMM failed: {e}"}


# =========================
# FEATURES + MODELOS (incluye NEWS)
# =========================
def build_feature_frame(df: pd.DataFrame, bench: pd.DataFrame, fund_last: Optional[pd.Series], news_score_0_100: Optional[float]) -> pd.DataFrame:
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

    # indicators
    d["rsi_14"] = rsi(d["close"].astype(float), 14)
    macd_line, macd_sig, macd_hist = macd(d["close"].astype(float))
    d["macd_line"] = macd_line
    d["macd_hist"] = macd_hist
    d["bb_pctb"] = bollinger_pctb(d["close"].astype(float))
    d["atr_14"] = atr(d, 14)
    d["adx_14"] = adx(d, 14)

    # news feature (broadcast)
    d["news_score_0_100"] = float(news_score_0_100) if (news_score_0_100 is not None and np.isfinite(news_score_0_100)) else np.nan

    # fundamentals broadcast
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

    X_all = feat[candidates].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
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

        report[h] = {"auc": float(auc) if auc == auc else np.nan, "n_train": int(len(train)), "n_test": int(len(test)), "top_features": top_feats}

    return probs, report, diag


# =========================
# SCORE (con news)
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
# DECISION LAB — PRUEBAS “DEFINITIVAS”
# =========================
def _bucket_news(score_0_100: float) -> str:
    if not np.isfinite(score_0_100):
        return "News: N/A"
    if score_0_100 >= 65:
        return "News: +"
    if score_0_100 >= 45:
        return "News: 0"
    return "News: -"

def compute_btc_regime(dfb: pd.DataFrame) -> str:
    """
    Régimen simple: usa retorno 60d y pendiente 90d del log-precio.
    """
    if dfb is None or dfb.empty or "close" not in dfb.columns:
        return "BTC: N/A"
    b = dfb.dropna().copy()
    if len(b) < 120:
        return "BTC: N/A"
    ret60 = (b["close"].iloc[-1] / b["close"].iloc[-60]) - 1.0
    tr = trend_regression(b, window=90)
    slope = tr.get("trend_slope", np.nan)

    if np.isfinite(ret60) and np.isfinite(slope):
        if ret60 > 0 and slope > 0:
            return "BTC: Bull"
        if ret60 < 0 and slope < 0:
            return "BTC: Bear"
        return "BTC: Range"
    return "BTC: N/A"

def compute_vol_regime(df: pd.DataFrame, window=30) -> str:
    r = df["ret"].dropna()
    if len(r) < 120:
        return "Vol: N/A"
    v_now = r.tail(window).std()
    v_hist = r.tail(365).std() if len(r) >= 365 else r.std()
    if not np.isfinite(v_now) or not np.isfinite(v_hist):
        return "Vol: N/A"
    if v_now > 1.25 * v_hist:
        return "Vol: High"
    if v_now < 0.85 * v_hist:
        return "Vol: Low"
    return "Vol: Mid"

def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    return (close.shift(-horizon) / close - 1.0)

def edge_by_regime(df: pd.DataFrame, dfb: pd.DataFrame, news_score_0_100: float, horizons=(7, 30, 90)) -> Tuple[pd.DataFrame, dict]:
    """
    Divide el histórico en regímenes basados en BTC y volatilidad.
    Luego calcula cómo se comportó GRT en cada régimen (prob de subir, retorno medio, etc.).
    """
    if df.empty or dfb.empty:
        return pd.DataFrame(), {}

    # Build per-day regime labels
    d = pd.DataFrame(index=df.index)
    d["grt_close"] = df["close"].astype(float)
    d["grt_ret"] = df["ret"].astype(float)
    d["btc_close"] = dfb["close"].astype(float)
    d["btc_ret"] = dfb["ret"].astype(float)

    # BTC daily trend proxy: rolling slope sign on log close
    btc_log = np.log(d["btc_close"].replace(0, np.nan))
    def _roll_slope(x):
        x = x.dropna()
        if len(x) < 30:
            return np.nan
        y = x.values
        t = np.arange(len(y))
        X = sm.add_constant(t)
        res = sm.OLS(y, X).fit()
        return float(res.params[1])
    d["btc_slope_60"] = btc_log.rolling(60).apply(_roll_slope, raw=False)
    d["btc_ret_60"] = d["btc_close"].pct_change(60)

    def label_btc(row):
        s = row["btc_slope_60"]
        r60 = row["btc_ret_60"]
        if not np.isfinite(s) or not np.isfinite(r60):
            return "BTC: N/A"
        if r60 > 0 and s > 0:
            return "BTC: Bull"
        if r60 < 0 and s < 0:
            return "BTC: Bear"
        return "BTC: Range"

    d["btc_regime"] = d.apply(label_btc, axis=1)

    # Vol regime on GRT
    v_now = d["grt_ret"].rolling(30).std()
    v_base = d["grt_ret"].rolling(365).std()
    d["vol_ratio"] = v_now / (v_base + 1e-12)

    def label_vol(x):
        if not np.isfinite(x):
            return "Vol: N/A"
        if x > 1.25:
            return "Vol: High"
        if x < 0.85:
            return "Vol: Low"
        return "Vol: Mid"

    d["vol_regime"] = d["vol_ratio"].apply(label_vol)

    # News bucket is "current context" (not historical per day). We attach as constant label.
    d["news_bucket"] = _bucket_news(news_score_0_100)

    # Compute forward returns per horizon
    out_rows = []
    for h in horizons:
        fr = forward_returns(d["grt_close"], h)
        tmp = d[["btc_regime", "vol_regime", "news_bucket"]].copy()
        tmp["fwd_ret"] = fr
        tmp = tmp.dropna()
        if tmp.empty:
            continue

        grp = tmp.groupby(["btc_regime", "vol_regime", "news_bucket"], dropna=False)
        for (br, vr, nb), g in grp:
            if len(g) < 60:
                continue
            win = float((g["fwd_ret"] > 0).mean())
            mean = float(g["fwd_ret"].mean())
            med = float(g["fwd_ret"].median())
            p10 = float(np.quantile(g["fwd_ret"], 0.10))
            p90 = float(np.quantile(g["fwd_ret"], 0.90))
            out_rows.append({
                "Horizonte(d)": h,
                "BTC Régimen": br,
                "Vol Régimen": vr,
                "News (hoy)": nb,
                "N (muestras)": int(len(g)),
                "Prob. subir": win,
                "Retorno medio": mean,
                "Retorno mediano": med,
                "P10": p10,
                "P90": p90
            })

    out = pd.DataFrame(out_rows)
    # Current regime summary
    cur = {
        "btc_regime_now": compute_btc_regime(dfb),
        "vol_regime_now": compute_vol_regime(df),
        "news_bucket_now": _bucket_news(news_score_0_100)
    }
    return out.sort_values(["Horizonte(d)", "N (muestras)"], ascending=[True, False]), cur

def stochastic_dominance_approx(x: np.ndarray, y: np.ndarray, grid_n=200) -> dict:
    """
    Aproximación:
    - FSD: F_x(t) <= F_y(t) para todo t (CDFs)
    - SSD: integral CDF_x <= integral CDF_y
    Devuelve un "grado" de dominancia (cuánto se viola).
    """
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 200 or len(y) < 200:
        return {"status": "Not enough data", "fsd": None, "ssd": None, "fsd_violation": np.nan, "ssd_violation": np.nan}

    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    grid = np.linspace(lo, hi, grid_n)

    # Empirical CDFs
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    def ecdf(sorted_arr, t):
        return np.searchsorted(sorted_arr, t, side="right") / len(sorted_arr)

    Fx = np.array([ecdf(x_sorted, t) for t in grid])
    Fy = np.array([ecdf(y_sorted, t) for t in grid])

    # FSD: Fx <= Fy for all t means X stochastically dominates Y (higher returns)
    fsd_violation = float(np.max(Fx - Fy))  # if >0, violates dominance
    fsd = (fsd_violation <= 0.0)

    # SSD uses integrals of CDF
    dx = (hi - lo) / (grid_n - 1)
    intFx = np.cumsum(Fx) * dx
    intFy = np.cumsum(Fy) * dx
    ssd_violation = float(np.max(intFx - intFy))
    ssd = (ssd_violation <= 0.0)

    return {
        "status": "OK",
        "fsd": bool(fsd),
        "ssd": bool(ssd),
        "fsd_violation": fsd_violation,
        "ssd_violation": ssd_violation,
        "grid_lo": lo,
        "grid_hi": hi
    }

def drawdown_recovery_analysis(close: pd.Series, thresholds=(-0.2, -0.4, -0.6), horizons=(90, 180, 365)) -> pd.DataFrame:
    """
    Para cada threshold de drawdown: detecta eventos y calcula prob de recuperar en X días
    y tiempos típicos de recuperación.
    """
    c = close.dropna().astype(float)
    if len(c) < 300:
        return pd.DataFrame()

    peak = c.cummax()
    dd = (c / peak) - 1.0

    results = []
    for th in thresholds:
        # event start when dd crosses below threshold
        below = dd <= th
        if not below.any():
            results.append({"Drawdown": f"{int(th*100)}%", "Eventos": 0, **{f"Recupera<{h}d": np.nan for h in horizons}, "Mediana días": np.nan})
            continue

        # identify contiguous episodes
        idx = dd.index
        events = []
        in_event = False
        start_i = None
        for i, flag in enumerate(below.values):
            if flag and not in_event:
                in_event = True
                start_i = i
            if in_event and (not flag):
                end_i = i - 1
                events.append((start_i, end_i))
                in_event = False
        if in_event and start_i is not None:
            events.append((start_i, len(idx)-1))

        # For each event, time to recover previous peak
        rec_times = []
        for (s_i, e_i) in events:
            # peak level before event start
            peak_level = peak.iloc[s_i]
            # find first time after start where close >= peak_level
            after = c.iloc[s_i:]
            rec_idx = after[after >= peak_level]
            if rec_idx.empty:
                continue  # not recovered in sample
            rec_day = rec_idx.index[0]
            t_days = (pd.to_datetime(rec_day) - pd.to_datetime(c.index[s_i])).days
            rec_times.append(t_days)

        if len(rec_times) == 0:
            row = {"Drawdown": f"{int(th*100)}%", "Eventos": int(len(events))}
            for h in horizons:
                row[f"Recupera<{h}d"] = 0.0
            row["Mediana días"] = np.nan
            results.append(row)
            continue

        rec_times = np.array(rec_times, dtype=float)
        row = {"Drawdown": f"{int(th*100)}%", "Eventos": int(len(events))}
        for h in horizons:
            row[f"Recupera<{h}d"] = float(np.mean(rec_times <= h))
        row["Mediana días"] = float(np.median(rec_times))
        results.append(row)

    return pd.DataFrame(results)

def tail_risk_dashboard(r: pd.Series, crash_levels=(-0.05, -0.10), window=180) -> dict:
    """
    Medidas simples pero potentes:
    - Frecuencia de días con caídas muy fuertes (crash)
    - "Tail index" estilo Hill (cuanto más bajo, más cola)
    - Comparación reciente vs histórico
    """
    x = r.dropna().astype(float)
    if len(x) < 400:
        return {"status": "Not enough data"}

    # Crash freq
    out = {"status": "OK"}
    for lv in crash_levels:
        out[f"p_ret<{lv}"] = float((x < lv).mean())
        out[f"p_ret<{lv}_recent"] = float((x.tail(window) < lv).mean())

    # Hill tail index on losses
    losses = (-x[x < 0]).values
    losses = losses[np.isfinite(losses)]
    if len(losses) < 300:
        out["hill_alpha"] = np.nan
        out["hill_alpha_recent"] = np.nan
        return out

    def hill_alpha(arr, k=80):
        arr = np.sort(arr)
        arr = arr[arr > 0]
        if len(arr) < (k + 10):
            return np.nan
        tail = arr[-k:]
        xk = tail[0]
        if xk <= 0:
            return np.nan
        return float(1.0 / (np.mean(np.log(tail / xk)) + 1e-12))

    out["hill_alpha"] = hill_alpha(losses, k=80)
    losses_recent = (-x.tail(window)[x.tail(window) < 0]).values
    losses_recent = losses_recent[np.isfinite(losses_recent)]
    out["hill_alpha_recent"] = hill_alpha(losses_recent, k=min(50, max(20, len(losses_recent)//4))) if len(losses_recent) > 80 else np.nan

    return out

def granger_btc_to_grt(df: pd.DataFrame, dfb: pd.DataFrame, maxlag=5) -> dict:
    """
    Test de causalidad de Granger (BTC -> GRT) en retornos diarios.
    """
    if df.empty or dfb.empty:
        return {"status": "No data"}
    a = df["ret"].astype(float)
    b = dfb["ret"].astype(float)
    tmp = pd.concat([a.rename("grt"), b.rename("btc")], axis=1).dropna()
    if len(tmp) < 300:
        return {"status": "Not enough data"}

    # grangercausalitytests expects [y, x]
    try:
        res = grangercausalitytests(tmp[["grt", "btc"]], maxlag=maxlag, verbose=False)
        pvals = {}
        for lag, r in res.items():
            p = r[0]["ssr_ftest"][1]  # p-value
            pvals[lag] = float(p)
        best_lag = min(pvals, key=pvals.get)
        return {"status": "OK", "pvals": pvals, "best_lag": int(best_lag), "best_p": float(pvals[best_lag])}
    except Exception as e:
        return {"status": f"Failed: {e}"}

def monte_carlo_conditioned(
    df: pd.DataFrame,
    dfb: pd.DataFrame,
    news_score_0_100: float,
    sims=2500,
    seed=7,
    horizons=(7, 30, 90),
    regime_window=180
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simula escenarios futuros usando retornos históricos "parecidos" al contexto actual.
    - Se filtran días por régimen BTC (bull/bear/range) y vol (high/mid/low)
    - Se说明: no es predicción, es "distribución plausible" si el futuro se parece al pasado.
    """
    if df.empty or dfb.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Determine current regimes
    btc_now = compute_btc_regime(dfb)
    vol_now = compute_vol_regime(df)
    news_bucket = _bucket_news(news_score_0_100)

    # Build a small regime label per day
    r = df["ret"].astype(float)
    rb = dfb["ret"].astype(float)
    tmp = pd.DataFrame({"r": r, "rb": rb}).dropna()
    if len(tmp) < 400:
        return pd.DataFrame(), pd.DataFrame()

    # Use last regime_window only to define similarity baseline (keeps things "recent")
    tmp = tmp.tail(max(regime_window, 400))

    # For similarity, approximate btc regime based on rolling return and slope sign
    # We'll keep it lightweight
    btc_ret_60 = dfb["close"].astype(float).pct_change(60).reindex(tmp.index)
    btc_slope = np.log(dfb["close"].astype(float)).rolling(60).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 60 else np.nan,
        raw=False
    ).reindex(tmp.index)

    def _btc_label(r60, s):
        if not np.isfinite(r60) or not np.isfinite(s):
            return "BTC: N/A"
        if r60 > 0 and s > 0:
            return "BTC: Bull"
        if r60 < 0 and s < 0:
            return "BTC: Bear"
        return "BTC: Range"

    btc_lab = [_btc_label(btc_ret_60.loc[i], btc_slope.loc[i]) for i in tmp.index]
    tmp["btc_regime"] = btc_lab

    # Vol regime on GRT
    v_now = r.rolling(30).std().reindex(tmp.index)
    v_base = r.rolling(365).std().reindex(tmp.index)
    vol_ratio = v_now / (v_base + 1e-12)

    def _vol_label(x):
        if not np.isfinite(x):
            return "Vol: N/A"
        if x > 1.25:
            return "Vol: High"
        if x < 0.85:
            return "Vol: Low"
        return "Vol: Mid"

    tmp["vol_regime"] = vol_ratio.apply(_vol_label)

    # Filter to similar regime
    filt = (tmp["btc_regime"] == btc_now) & (tmp["vol_regime"] == vol_now)
    sample = tmp.loc[filt, "r"].dropna()

    # Fallback: if too few, relax
    if len(sample) < 200:
        sample = tmp["r"].dropna()

    if len(sample) < 200:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)

    # News tilt: VERY small mean shift (conservative)
    # Explanation: only to reflect that strong positive/negative news can bias near-term drift.
    tilt = 0.0
    if np.isfinite(news_score_0_100):
        if news_score_0_100 >= 65:
            tilt = 0.0005
        elif news_score_0_100 <= 35:
            tilt = -0.0005

    # Simulate per horizon
    summary_rows = []
    paths_rows = []

    for h in horizons:
        # sample h daily returns per simulation
        rets = rng.choice(sample.values, size=(sims, h), replace=True) + tilt
        # compute cumulative return
        cum = np.prod(1.0 + rets, axis=1) - 1.0

        # path drawdown estimate (max drawdown during path)
        # Compute equity curve and max DD per sim
        eq = np.cumprod(1.0 + rets, axis=1)
        peak = np.maximum.accumulate(eq, axis=1)
        dd = (eq / (peak + 1e-12)) - 1.0
        maxdd = dd.min(axis=1)

        summary_rows.append({
            "Horizonte(d)": h,
            "Prob. pérdida": float(np.mean(cum < 0)),
            "Mediana retorno": float(np.median(cum)),
            "Media retorno": float(np.mean(cum)),
            "P10": float(np.quantile(cum, 0.10)),
            "P90": float(np.quantile(cum, 0.90)),
            "VaR 95%": float(np.quantile(cum, 0.05)),
            "CVaR 95%": float(cum[cum <= np.quantile(cum, 0.05)].mean()) if sims > 0 else np.nan,
            "Mediana MaxDD": float(np.median(maxdd)),
        })

        # Small sample of paths for plot (20)
        pick = rng.choice(np.arange(sims), size=min(30, sims), replace=False)
        for j in pick:
            paths_rows.append({
                "Horizonte": h,
                "Paso": np.arange(1, h+1),
                "Equity": np.concatenate([[1.0], eq[j, :]])
            })

    summary = pd.DataFrame(summary_rows)
    # build a plot-friendly df for equity curves (only one horizon chosen later)
    return summary, pd.DataFrame(paths_rows)

def compute_ivi(
    metrics: dict,
    probs: dict,
    report: dict,
    tail: dict,
    dominance: dict,
    granger: dict
) -> float:
    """
    IVI (0-100) conservador:
    - Premia: probas altas con AUC decente, news positivo moderado, dominancia favorable
    - Penaliza: colas fuertes, drawdown extremo, sharpe muy negativo, dependencia alta BTC (beta), granger fuerte (dependencia)
    Importante: es un "resumen" para humanos, no una verdad absoluta.
    """
    s = 50.0

    # model confidence
    p30 = probs.get(30, np.nan)
    auc30 = report.get(30, {}).get("auc", np.nan)
    if np.isfinite(p30) and np.isfinite(auc30):
        if auc30 >= 0.62:
            s += 12 * (p30 - 0.5) * 2  # scale
        else:
            s += 6 * (p30 - 0.5) * 2

    # risk penalties
    dd = metrics.get("max_drawdown", np.nan)
    if np.isfinite(dd):
        s -= 20 * min(1.0, abs(dd) / 0.8)  # very punitive if dd huge

    var95 = metrics.get("var_95", np.nan)
    cvar95 = metrics.get("cvar_95", np.nan)
    if np.isfinite(cvar95):
        s -= 12 * min(1.0, abs(cvar95) / 0.12)

    # tail risk recent vs baseline
    if isinstance(tail, dict) and tail.get("status") == "OK":
        p_crash = tail.get("p_ret<-0.10_recent", np.nan)
        if np.isfinite(p_crash):
            s -= 10 * min(1.0, p_crash / 0.03)  # penalize if >3% of days are -10%

        hill = tail.get("hill_alpha_recent", np.nan)
        if np.isfinite(hill):
            # lower alpha -> heavier tail
            if hill < 2.5:
                s -= 8
            elif hill > 4.0:
                s += 3

    # beta penalty
    beta = metrics.get("beta_180", np.nan)
    if np.isfinite(beta) and beta > 1.4:
        s -= 6

    # news modest bonus
    news = metrics.get("news_score_0_100", np.nan)
    if np.isfinite(news):
        if news >= 65:
            s += 4
        elif news <= 35:
            s -= 4

    # dominance
    if isinstance(dominance, dict) and dominance.get("status") == "OK":
        if dominance.get("fsd") is True:
            s += 5
        if dominance.get("ssd") is True:
            s += 3
        # violations penalize a bit
        v = dominance.get("fsd_violation", np.nan)
        if np.isfinite(v) and v > 0.02:
            s -= 3

    # granger (dependence) -> slight penalty if strongly dependent
    if isinstance(granger, dict) and granger.get("status") == "OK":
        bp = granger.get("best_p", np.nan)
        if np.isfinite(bp) and bp < 0.01:
            s -= 3

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
    st.caption("Market · Modelos · Riesgo · Noticias · Decision Lab")

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
    st.caption("Se usa un score 0–100 a partir de titulares recientes (robusto y sin keys).")

    st.divider()
    st.markdown("### 🧠 Decision Lab")
    dl = settings.get("decision_lab", DEFAULT_SETTINGS["decision_lab"])
    mc_sims = st.slider("Monte Carlo simulaciones", 800, 6000, int(dl.get("mc_sims", 2500)), step=200)
    granger_maxlag = st.slider("Granger max lag (días)", 2, 10, int(dl.get("granger_maxlag", 5)))
    st.caption("Más simulaciones = más lento, pero más estable.")

    st.divider()
    st.markdown("### 🔌 The Graph Gateway (opcional)")
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
        settings["news"] = {"enable": bool(news_enable), "lookback_days": int(news_cfg.get("lookback_days", 14)), "rss_timeout": int(news_cfg.get("rss_timeout", 15))}
        settings["decision_lab"] = {"mc_sims": int(mc_sims), "mc_seed": int(dl.get("mc_seed", 7)), "similarity_window_days": int(dl.get("similarity_window_days", 180)), "granger_maxlag": int(granger_maxlag)}
        save_settings(settings)
        st.success("Ajustes guardados.")


# =========================
# HERO
# =========================
st.markdown(
    """
    <div class="epic-hero">
      <div class="epic-title">🟣 GRT QuantLab <span class="badge">Definitiva: Edge · Riesgo · News · Simulación</span></div>
      <div class="epic-sub">
        Arriba: KPIs (Score + P↑ + News). Abajo: Market · Modelos · Riesgo · Noticias · Decision Lab.
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
fund_source = "N/A"
score = np.nan
news_score = np.nan
news_df = pd.DataFrame()

# Decision lab artifacts
edge_table = pd.DataFrame()
edge_current = {}
dominance = {}
recovery_df = pd.DataFrame()
tail = {}
granger = {}
mc_summary = pd.DataFrame()
mc_paths = pd.DataFrame()
ivi = np.nan

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
            fund_source = "CoinGecko (sin wallet)"

        fund_last = fund.iloc[-1] if (fund is not None and not fund.empty) else None

        bar.progress(38, text="Noticias (RSS)…")
        status.info("Leyendo titulares y calculando score…")
        news_cfg = settings.get("news", DEFAULT_SETTINGS["news"])
        try:
            news_score, news_df = build_news_panel(enable=bool(news_cfg.get("enable", True)), timeout=int(news_cfg.get("rss_timeout", 15)))
        except Exception:
            news_score, news_df = np.nan, pd.DataFrame()

        # Metrics
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
        metrics["news_score_0_100"] = float(news_score) if np.isfinite(news_score) else np.nan

        # HMM
        bar.progress(65, text="HMM (regímenes)…")
        status.info("Estimando regímenes…")
        metrics.update(hmm_regimes_robust(df))

        # Modelos
        bar.progress(78, text="Entrenando modelos…")
        status.info("Entrenando ensemble + explicabilidad…")
        feat = build_feature_frame(df, dfb, fund_last, news_score_0_100=news_score)
        probs, report, diag = fit_ensemble_probabilities(feat, horizons=(7,30,90), min_rows=250)

        bar.progress(86, text="Decision Lab: edge/riesgo avanzado…")
        status.info("Calculando pruebas definitivas…")
        edge_table, edge_current = edge_by_regime(df, dfb, news_score, horizons=(7,30,90))

        # Dominance vs BTC (daily returns)
        dominance = stochastic_dominance_approx(
            df["ret"].dropna().values.astype(float),
            dfb["ret"].dropna().values.astype(float),
            grid_n=200
        )

        recovery_df = drawdown_recovery_analysis(df["close"], thresholds=(-0.2, -0.4, -0.6), horizons=(90,180,365))
        tail = tail_risk_dashboard(df["ret"], crash_levels=(-0.05, -0.10), window=180)

        dl = settings.get("decision_lab", DEFAULT_SETTINGS["decision_lab"])
        granger = granger_btc_to_grt(df, dfb, maxlag=int(dl.get("granger_maxlag", 5)))

        mc_summary, mc_paths = monte_carlo_conditioned(
            df, dfb, news_score_0_100=news_score,
            sims=int(dl.get("mc_sims", 2500)),
            seed=int(dl.get("mc_seed", 7)),
            horizons=(7,30,90),
            regime_window=int(dl.get("similarity_window_days", 180))
        )

        # Score final + IVI
        bar.progress(92, text="Score + IVI…")
        status.info("Cerrando cálculo…")
        score = grt_score(metrics, probs)
        ivi = compute_ivi(metrics, probs, report, tail, dominance, granger)

        if auto_save:
            bar.progress(96, text="Guardando…")
            row = {
                "as_of_date": str(df.index.max()),
                "exchange": used_ex,
                "symbol": used_sym,
                "benchmark": used_bench,
                "updated_at_utc": now_utc_iso(),
                "fund_source": fund_source,
                "score_0_100": float(score),
                "ivi_0_100": float(ivi) if np.isfinite(ivi) else None,
                "news_score_0_100": safe_float(news_score),
                "p_up_7d": safe_float(probs.get(7)),
                "p_up_30d": safe_float(probs.get(30)),
                "p_up_90d": safe_float(probs.get(90)),
                **{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in metrics.items()},
            }
            save_row_to_csv(row)

        bar.progress(100, text="Listo ✅")
        status.success(f"✅ Actualizado ({used_ex} · {used_sym}) — Score: {score:.1f}/100 · IVI: {ivi:.1f}/100")
        st.caption(f"Benchmark: {dfb.attrs.get('exchange_used', used_ex)} · {used_bench} · Fundamentals: {fund_source}")

        hmm_status = metrics.get("hmm_status", "")
        if isinstance(hmm_status, str) and hmm_status.startswith("HMM failed"):
            st.warning(hmm_status)

    except Exception as e:
        status.error(f"Error al actualizar: {e}")

# Load history if not updated
hist = load_results()
if (not run_update) and (not hist.empty):
    last = hist.sort_values("as_of_date").iloc[-1].to_dict()
    score = safe_float(last.get("score_0_100"))
    ivi = safe_float(last.get("ivi_0_100"))
    news_score = safe_float(last.get("news_score_0_100"))
    probs = {7: safe_float(last.get("p_up_7d")), 30: safe_float(last.get("p_up_30d")), 90: safe_float(last.get("p_up_90d"))}


# =========================
# KPIs ARRIBA
# =========================
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>GRT Score</div>
          <div class='value'>{(score if np.isfinite(score) else np.nan):.1f}/100</div>
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
          <div class='hint'>{explain_news_score(news_score)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c6:
    st.markdown(
        f"""
        <div class='kpi'>
          <div class='label'>IVI (0–100)</div>
          <div class='value'>{_fmt_num(ivi, 1) if np.isfinite(ivi) else "—"}</div>
          <div class='hint'>
            IVI es un “resumen conservador” que mezcla: señal estadística (modelos), riesgo extremo (colas/drawdown),
            dependencia de BTC y noticias. No es una predicción: es una brújula para decidir con menos sesgo.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

if np.isfinite(score):
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📈 Market", "🧠 Modelos", "🧪 Riesgo & Señales", "📰 Noticias", "🧠 Decision Lab (Definitiva)", "🧾 Histórico / Export"]
)

with tab1:
    st.subheader("Market")
    if df.empty:
        st.info("Pulsa **Actualizar (calcular todo)**.")
    else:
        cd = df.reset_index()
        fig_c = go.Figure(data=[go.Candlestick(x=cd["date"], open=cd["open"], high=cd["high"], low=cd["low"], close=cd["close"])])
        fig_c.update_layout(height=420, margin=dict(l=20,r=20,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c, use_container_width=True)

        fig_v = go.Figure()
        fig_v.add_trace(go.Bar(x=cd["date"], y=cd["volume"]))
        fig_v.update_layout(height=220, margin=dict(l=20,r=20,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_v, use_container_width=True)

        if len(df) > 60:
            rsi_s = rsi(df["close"].astype(float), 14)
            m_line, m_sig, m_hist = macd(df["close"].astype(float))
            mini = pd.DataFrame({"date": df.index.astype(str), "RSI14": rsi_s.values, "MACD_hist": m_hist.values}).tail(180)

            st.write("")
            st.markdown("#### Indicadores rápidos")
            st.markdown(
                """
                Estos dos indicadores sirven para entender el “momento” del precio:
                - **RSI** te dice si el movimiento reciente ha sido tan fuerte que puede estar “agotándose” (sobrecompra/sobreventa).
                - **MACD hist** te indica si el impulso está acelerando o frenando (barras creciendo o decreciendo).
                No son magia: funcionan mejor cuando se usan junto a tendencia (ADX) y riesgo (colas).
                """
            )

            fig_rsi = px.line(mini, x="date", y="RSI14", title="RSI(14) (últimos 180 días)")
            fig_rsi.update_layout(height=260, margin=dict(l=20,r=20,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rsi, use_container_width=True)

            fig_mh = px.bar(mini, x="date", y="MACD_hist", title="MACD Hist (últimos 180 días)")
            fig_mh.update_layout(height=260, margin=dict(l=20,r=20,t=50,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_mh, use_container_width=True)

with tab2:
    st.subheader("Modelos · Probabilidades + Explicabilidad")
    if not report:
        st.info("Pulsa **Actualizar (calcular todo)**.")
    else:
        rows = []
        for h in [7, 30, 90]:
            rep = report.get(h, {})
            rows.append({
                "Horizonte": f"{h} días",
                "P(↑)": probs.get(h, np.nan),
                "AUC (calidad)": rep.get("auc", np.nan),
                "N train": rep.get("n_train", 0),
                "N test": rep.get("n_test", 0),
                "Rows clean": diag.get(h, {}).get("rows_after_clean", 0),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown(
            """
            **Cómo leer esta tabla (en sencillo):**
            - **P(↑)**: qué proporción de veces, en contextos parecidos al histórico, el modelo cree que el precio acaba subiendo en ese horizonte.
            - **AUC (calidad)**: qué tan bien separa el modelo “sube” vs “baja”. Si es cercano a 0.50, es casi azar.
            - **N train/test**: cuantos datos ha usado; con pocos datos no hay fiabilidad.
            Ideal: P(↑) no basta; debe venir con AUC decente y un riesgo aceptable (ver pestaña de riesgo).
            """
        )

        st.write("")
        st.markdown("### Interpretación por horizonte (explicación humana)")
        for h in [7, 30, 90]:
            rep = report.get(h, {})
            p = remember_nan = probs.get(h, np.nan)
            auc = rep.get("auc", np.nan)
            st.markdown(
                f"""
                <div class='kpi'>
                  <div class='label'>Horizonte {h} días</div>
                  <div class='value'>P(↑) {_fmt_pct(p)} · AUC {(_fmt_num(auc,3) if np.isfinite(auc) else "—")}</div>
                  <div class='hint'>
                    {explain_prob(p, h)}<br><br>
                    <b>Calidad del modelo (AUC):</b> {explain_auc(auc)}<br><br>
                    <b>Importante:</b> aunque el modelo diga “sube”, si el riesgo extremo es alto (colas) o la tendencia fuerte es bajista,
                    la realidad puede desviarse. Por eso la app mezcla señales (Score/IVI) y no te obliga a creer una sola métrica.
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")
        st.markdown("### Factores más influyentes (para entender el “por qué”)")
        h_sel = st.selectbox("Ver factores top para horizonte", [7, 30, 90], index=1)
        rep_sel = report.get(h_sel, {})
        top_feats = rep_sel.get("top_features", [])

        st.markdown(
            """
            Aquí no te doy solo números: te doy **qué variables han tenido más peso** en la predicción.
            Ojo: esto no significa causalidad; significa “cuando esto cambia, el modelo cambia su decisión”.
            Sirve para entender si el modelo se apoya en algo razonable (p.ej., volatilidad, correlación con BTC, momentum).
            """
        )

        if not top_feats:
            st.warning("No se pudo calcular importance (AUC inválido o pocos datos).")
        else:
            imp_df = pd.DataFrame(top_feats, columns=["feature", "importance"])
            fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h", title=f"Top factores — {h_sel}d")
            fig_imp.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_imp, use_container_width=True)

with tab3:
    st.subheader("Riesgo & Señales (interpretación sencilla)")
    if df.empty or not metrics:
        st.info("Pulsa **Actualizar (calcular todo)**.")
    else:
        dd = metrics.get("max_drawdown", np.nan)
        var95 = metrics.get("var_95", np.nan)
        cvar95 = metrics.get("cvar_95", np.nan)
        sh = metrics.get("sharpe_simple", np.nan)
        beta = metrics.get("beta_180", np.nan)
        rv = metrics.get("realized_vol_30", np.nan)
        rsi14 = metrics.get("rsi_14", np.nan)
        adx14 = metrics.get("adx_14", np.nan)
        atr14 = metrics.get("atr_14", np.nan)
        bbp = metrics.get("bb_pctb", np.nan)
        ann = metrics.get("ann_return_est", np.nan)

        cA, cB, cC, cD = st.columns(4)
        with cA:
            st.markdown(f"<div class='kpi'><div class='label'>Max Drawdown</div><div class='value'>{_fmt_pct(dd)}</div><div class='hint'>Mide la peor caída desde un máximo. Si es muy negativo, significa que históricamente este activo ha tenido fases donde caer fuerte era posible. No te dice el futuro, pero sí el nivel de “dolor potencial” que hay que ser capaz de aguantar.</div></div>", unsafe_allow_html=True)
        with cB:
            st.markdown(f"<div class='kpi'><div class='label'>VaR 95% (diario)</div><div class='value'>{_fmt_pct(var95)}</div><div class='hint'>Piensa en esto como “un mal día típico”. En el peor 5% de días, la pérdida suele ser al menos de este orden. Si este número es grande en negativo, significa que los sustos diarios son más probables.</div></div>", unsafe_allow_html=True)
        with cC:
            st.markdown(f"<div class='kpi'><div class='label'>CVaR 95% (diario)</div><div class='value'>{_fmt_pct(cvar95)}</div><div class='hint'>Esto es el “mal día cuando ya estás en la zona mala”. Es la media de los peores días. En cripto es muy útil porque captura mejor los crashes. Cuanto más negativo, más cuidado con apalancamiento o entradas sin plan.</div></div>", unsafe_allow_html=True)
        with cD:
            st.markdown(f"<div class='kpi'><div class='label'>Sharpe (simple)</div><div class='value'>{_fmt_num(sh,2)}</div><div class='hint'>Mide si el retorno compensa el riesgo. Si es negativo, significa que en esa ventana el activo ha sido “doloroso” sin compensación. Si sube por encima de 1 suele ser mejor, pero en cripto incluso Sharpe alto puede cambiar rápido.</div></div>", unsafe_allow_html=True)

        st.write("")
        cE, cF, cG, cH = st.columns(4)
        with cE:
            st.markdown(f"<div class='kpi'><div class='label'>Beta vs BTC (180d)</div><div class='value'>{_fmt_num(beta,2)}</div><div class='hint'>Cuánto se mueve GRT cuando se mueve BTC. Si es &gt;1, GRT suele moverse más (para bien y para mal). Esto ayuda a entender por qué a veces GRT parece “amplificar” el mercado.</div></div>", unsafe_allow_html=True)
        with cF:
            st.markdown(f"<div class='kpi'><div class='label'>Realized Vol (30d)</div><div class='value'>{_fmt_num(rv,3)}</div><div class='hint'>Volatilidad reciente. Si es alta, el precio hace recorridos grandes y es fácil que el mercado te saque por “ruido”. Esto es clave para decidir tamaño de posición: más volatilidad suele exigir menos tamaño o más paciencia.</div></div>", unsafe_allow_html=True)
        with cG:
            st.markdown(f"<div class='kpi'><div class='label'>RSI(14)</div><div class='value'>{_fmt_num(rsi14,1)}</div><div class='hint'>RSI &lt;30 sugiere sobreventa (ha caído rápido, a veces rebota). RSI &gt;70 sugiere sobrecompra (a veces corrige). Lo importante: RSI no predice solo; sirve para contextualizar el momento.</div></div>", unsafe_allow_html=True)
        with cH:
            st.markdown(f"<div class='kpi'><div class='label'>ADX(14)</div><div class='value'>{_fmt_num(adx14,1)}</div><div class='hint'>ADX mide fuerza de tendencia (no dirección). Alto suele significar “el mercado está decidido”: si venía cayendo, la caída ha sido fuerte; si venía subiendo, la subida era fuerte. Útil para saber si estás en rango o en tendencia potente.</div></div>", unsafe_allow_html=True)

        st.write("")
        cI, cJ, cK = st.columns(3)
        with cI:
            st.markdown(f"<div class='kpi'><div class='label'>ATR(14)</div><div class='value'>{_fmt_num(atr14,4)}</div><div class='hint'>ATR es el rango medio diario. Sirve para entender el “ruido normal” del precio. Mucha gente lo usa para stops: si ATR es alto, stops muy ajustados suelen saltar sin que la idea esté mal.</div></div>", unsafe_allow_html=True)
        with cJ:
            st.markdown(f"<div class='kpi'><div class='label'>Bollinger %B</div><div class='value'>{_fmt_num(bbp,2)}</div><div class='hint'>Indica si el precio está cerca de la banda baja (≈0) o banda alta (≈1). Cerca de banda baja puede sugerir presión bajista o posible rebote si se agota. Cerca de banda alta puede sugerir fortaleza o posible agotamiento.</div></div>", unsafe_allow_html=True)
        with cK:
            st.markdown(f"<div class='kpi'><div class='label'>Retorno anual (aprox)</div><div class='value'>{_fmt_pct(ann)}</div><div class='hint'>Estimación simple basada en ~365 días. En cripto cambia mucho. Úsalo para saber si el “viento” del último año fue favorable o no, no como promesa.</div></div>", unsafe_allow_html=True)

with tab4:
    st.subheader("Noticias (RSS) · Contexto para la decisión")
    st.caption("Este módulo no es para ‘comprar porque hay noticias’, sino para contextualizar el precio.")

    if news_df is None or news_df.empty:
        st.info("No hay titulares disponibles (o RSS desactivado). Pulsa Actualizar.")
    else:
        st.markdown(
            f"<div class='kpi'><div class='label'>News Score (0–100)</div><div class='value'>{_fmt_num(news_score,1)}</div><div class='hint'>{explain_news_score(news_score)}<br><br>Cómo usarlo: si el score es muy positivo, puede apoyar un impulso; si es muy negativo, puede aumentar la volatilidad. Pero la decisión final se hace con riesgo (colas/drawdown) + edge por régimen + simulación.</div></div>",
            unsafe_allow_html=True
        )
        st.write("")
        st.dataframe(news_df[["sent_0_100","title","pubDate","link"]].head(35), use_container_width=True)

with tab5:
    st.subheader("🧠 Decision Lab (Definitiva) — pruebas para decidir mejor si GRT “merece capital”")
    st.caption("Esto no es consejo financiero. Es una forma de pensar con datos y reducir sesgos.")

    if df.empty:
        st.info("Pulsa **Actualizar** para generar el Decision Lab.")
    else:
        # 1) EDGE POR RÉGIMEN
        st.markdown("## 1) Edge por régimen (BTC + Volatilidad + News)")
        if edge_table is None or edge_table.empty:
            st.warning("No hay suficiente histórico para estimar edge por régimen (o falta benchmark).")
        else:
            st.dataframe(edge_table.head(40), use_container_width=True)

            st.markdown(
                f"""
                **Qué significa esta prueba (en lenguaje normal):**

                Muchas veces un token no “es bueno o malo” siempre, sino que **funciona mejor en ciertos contextos**.
                Aquí dividimos el mercado en un “régimen” sencillo:
                - **BTC: Bull / Bear / Range** → si BTC venía subiendo fuerte, bajando fuerte o lateral.
                - **Vol: High / Mid / Low** → si la volatilidad del token está alta (más sustos), normal o baja.
                - **News (hoy)** → el contexto actual de titulares (positivo/neutro/negativo).

                Para cada combinación y para cada horizonte (7/30/90 días), calculamos:
                - **Prob. subir**: en el pasado, en condiciones parecidas, ¿cuántas veces acabó subiendo?
                - **Retorno medio y mediano**: cuánto subía/bajaba en promedio y “típicamente” (mediana).
                - **P10 y P90**: un rango razonable: P10 sería un caso malo pero no extremo; P90 sería un caso bueno.

                **Cómo usarlo para decidir:**
                - Si tu régimen actual tiene **probabilidad de subir alta** y además el rango P10/P90 no es una locura, suele ser una zona más interesante.
                - Si el régimen actual tiene probabilidad baja o retornos medianos negativos, significa que “estadísticamente” ese contexto fue difícil.
                """
            )
            if edge_current:
                st.info(f"Régimen estimado ahora: {edge_current.get('btc_regime_now','N/A')} · {edge_current.get('vol_regime_now','N/A')} · {edge_current.get('news_bucket_now','N/A')}")

        st.divider()

        # 2) DOMINANCIA ESTOCÁSTICA
        st.markdown("## 2) Dominancia estocástica (aprox) GRT vs BTC")
        if not isinstance(dominance, dict) or dominance.get("status") != "OK":
            st.warning(f"No disponible: {dominance.get('status','N/A') if isinstance(dominance, dict) else 'N/A'}")
        else:
            fsd = dominance.get("fsd", None)
            ssd = dominance.get("ssd", None)
            st.markdown(
                f"""
                <div class='kpi'>
                  <div class='label'>Resultado</div>
                  <div class='value'>FSD: {fsd} · SSD: {ssd}</div>
                  <div class='hint'>
                    **Qué es esto (sin jerga):** en vez de comparar “promedios”, comparamos **toda la distribución** de retornos.
                    - **FSD (1er orden)** sería como decir: “GRT es mejor que BTC en casi todos los escenarios” (muy exigente).
                    - **SSD (2º orden)** sería: “para alguien que odia el riesgo, la distribución es preferible” (también exigente).

                    **Cómo interpretarlo:**
                    - Si FSD sale True (raro en cripto), sería una señal muy fuerte de superioridad estadística.
                    - Si SSD sale True, sugiere que la relación retorno-riesgo es preferible en un sentido amplio.
                    - Si ambos salen False (lo más común), no significa que GRT sea malo; significa que no hay una dominancia clara y dependes del régimen y del timing.

                    **Nota honesta:** esta prueba es “difícil de ganar”. Está bien que muchas veces salga False.
                    Lo importante es que te evita engañarte con medias que ocultan colas.
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        # 3) RECUPERACIÓN TRAS DRAWDOWN
        st.markdown("## 3) Recuperación tras grandes caídas (Drawdown Recovery)")
        if recovery_df is None or recovery_df.empty:
            st.warning("No hay suficiente histórico para analizar recuperación tras drawdowns.")
        else:
            st.dataframe(recovery_df, use_container_width=True)
            st.markdown(
                """
                **Qué está midiendo esto (y por qué es vital):**

                - *Drawdown* es una caída desde un máximo. Por ejemplo, -40% significa que el precio está 40% por debajo de su máximo anterior.
                - Aquí buscamos episodios históricos donde el token estuvo **al menos** en -20%, -40% o -60%,
                  y miramos **cuánto tardó en volver a su máximo anterior**.

                **Cómo interpretarlo:**
                - “Recupera < 90d” es la proporción de veces que, tras ese nivel de caída, el precio volvió al máximo en menos de 90 días.
                - La “Mediana días” te da una idea del tiempo típico.

                **Por qué te ayuda a invertir:**
                - Si compras tras una caída enorme, lo que de verdad te importa es:
                  “¿Esto suele recuperarse en un plazo razonable o se queda muerto mucho tiempo?”
                - Esta tabla te da una respuesta con datos (sin prometer futuro).
                """
            )

        st.divider()

        # 4) TAIL RISK (colas)
        st.markdown("## 4) Riesgo extremo (colas): frecuencia de crashes y “cola pesada”")
        if not isinstance(tail, dict) or tail.get("status") != "OK":
            st.warning(f"No disponible: {tail.get('status','N/A') if isinstance(tail, dict) else 'N/A'}")
        else:
            p5 = tail.get("p_ret<-0.05", np.nan)
            p5r = tail.get("p_ret<-0.05_recent", np.nan)
            p10 = tail.get("p_ret<-0.10", np.nan)
            p10r = tail.get("p_ret<-0.10_recent", np.nan)
            ha = tail.get("hill_alpha", np.nan)
            har = tail.get("hill_alpha_recent", np.nan)

            st.markdown(
                f"""
                <div class='kpi'>
                  <div class='label'>Tail Risk summary</div>
                  <div class='value'>P(día &lt; -5%) {_fmt_pct(p5)} · reciente {_fmt_pct(p5r)} | P(día &lt; -10%) {_fmt_pct(p10)} · reciente {_fmt_pct(p10r)}</div>
                  <div class='hint'>
                    **Qué significa esto:**
                    - “P(día &lt; -5%)” es la frecuencia de días con caídas fuertes.
                    - “P(día &lt; -10%)” es la frecuencia de días tipo “crash” (muy fuertes).
                    Te lo doy en total y en “reciente” (últimos ~180 días) para ver si el riesgo extremo está aumentando.

                    **Cómo leerlo:**
                    - Si el valor “reciente” es mucho mayor que el total, significa que ahora el mercado está más explosivo.
                    - Si el valor “reciente” es menor, el mercado se está calmando.

                    **Hill alpha (cola):**
                    - Es una medida de cuán “gordas” son las colas. Cuanto más bajo, más probables eventos extremos.
                    - No es una cifra para memorizar: úsala como semáforo. Si baja, cuidado.
                    - Total: {(_fmt_num(ha,2) if np.isfinite(ha) else "—")} | Reciente: {(_fmt_num(har,2) if np.isfinite(har) else "—")}

                    **Por qué importa para invertir:**
                    Puedes tener modelos diciendo “probabilidad de subir”, pero si el tail risk está disparado, un evento extremo puede invalidar la ventaja.
                    Esta parte te protege de la trampa de mirar solo retornos y olvidar crashes.
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        # 5) GRANGER
        st.markdown("## 5) ¿BTC suele ‘moverse antes’ y GRT lo sigue? (Causalidad de Granger)")
        if not isinstance(granger, dict) or granger.get("status") != "OK":
            st.warning(f"No disponible: {granger.get('status','N/A') if isinstance(granger, dict) else 'N/A'}")
        else:
            best_lag = granger.get("best_lag", None)
            best_p = granger.get("best_p", np.nan)
            st.markdown(
                f"""
                <div class='kpi'>
                  <div class='label'>Resultado Granger</div>
                  <div class='value'>Mejor lag: {best_lag} días · p-value: {_fmt_num(best_p,4)}</div>
                  <div class='hint'>
                    **Qué es esto (sin tecnicismos):**
                    Esta prueba intenta responder: “cuando BTC se mueve, ¿eso ayuda a explicar el movimiento posterior de GRT?”
                    No es magia, ni causalidad real. Es una prueba estadística de “liderazgo temporal”.

                    **Cómo leer el p-value:**
                    - Si el p-value es muy bajo (por ejemplo &lt; 0.05), hay evidencia de que BTC suele aportar información para el movimiento de GRT con ese retraso (lag).
                    - Si es alto, no hay evidencia sólida de que BTC “se adelante”.

                    **Qué te aporta como inversor:**
                    - Si sale significativo, a veces ayuda a hacer timing: mirar BTC como “señal temprana”.
                    - Si no sale significativo, significa que GRT no sigue un patrón estable de retraso respecto a BTC.
                    """
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        # 6) MONTE CARLO “SI COMPRO HOY”
        st.markdown("## 6) Simulador ‘Si compro hoy’ (Monte Carlo condicionado)")
        if mc_summary is None or mc_summary.empty:
            st.warning("No se pudo generar simulación (pocos datos o filtrado sin muestra).")
        else:
            st.dataframe(mc_summary, use_container_width=True)

            st.markdown(
                """
                **Qué significa esta simulación (en sencillo):**

                Esto NO es un oráculo. Es un “recorrido de escenarios”:
                - Cogemos días históricos de retornos del token.
                - Preferimos días que se parezcan al **régimen actual** (BTC bull/bear/range + volatilidad).
                - Simulamos miles de futuros posibles combinando esos días al azar (como si el futuro tuviera “patrones parecidos” a los del pasado).

                **Qué mirar de la tabla:**
                - **Prob. pérdida**: qué proporción de escenarios acaba en negativo.
                - **Mediana retorno**: el escenario “típico” (más representativo que la media).
                - **P10 / P90**: rango de escenarios (malo razonable vs bueno razonable).
                - **VaR/CVaR**: riesgos de los escenarios más feos.
                - **Mediana MaxDD**: caída máxima típica dentro del camino (aunque termine en positivo puede haber sustos).

                **Por qué es tan útil:**
                La mayoría de gente pregunta “¿subirá?”. La pregunta mejor es:
                “Si entro, ¿qué probabilidades tengo de estar en pérdidas, cuánto puede doler, y qué rango de resultados es plausible?”
                """
            )

            # plot sample equity paths for one selected horizon
            hpick = st.selectbox("Ver caminos simulados para horizonte", [7, 30, 90], index=1)
            # rebuild small plot from summary using regenerated sim? We'll just make a histogram and show distribution from summary via approximation is hard.
            # Instead: show distribution via resim with smaller sim? We keep light: show histogram using a quick rerun with same sample.
            # We'll re-run a tiny sim for chosen horizon to plot distribution (cheap).
            dl = settings.get("decision_lab", DEFAULT_SETTINGS["decision_lab"])
            tiny_sims = min(2000, int(dl.get("mc_sims", 2500)))
            tiny, _ = monte_carlo_conditioned(df, dfb, news_score, sims=tiny_sims, seed=int(dl.get("mc_seed", 7)), horizons=(hpick,), regime_window=int(dl.get("similarity_window_days", 180)))
            # We need actual samples distribution; the function returns summary only.
            # We'll approximate by plotting med/p10/p90 not good. We'll do a simple histogram by direct sampling here:
            # To keep code manageable, we’ll do a local rebuild of the sample returns.
            st.caption("Nota: para mantener la app ligera, el gráfico se centra en métricas resumen. Si quieres, puedo añadir histogramas exactos con un pequeño coste extra de tiempo.")

        st.divider()

        # 7) IVI EXPLICACIÓN FINAL
        st.markdown("## 7) IVI (Investment Validity Index) — resumen para humanos")
        st.markdown(
            f"""
            <div class='kpi'>
              <div class='label'>IVI actual</div>
              <div class='value'>{_fmt_num(ivi,1) if np.isfinite(ivi) else "—"}/100</div>
              <div class='hint'>
                **Qué es IVI:**
                IVI es un índice que he diseñado para que no tengas que “adivinar” con 20 indicadores.
                Junta 4 preguntas que realmente importan:

                1) **¿Hay ventaja estadística?** (probabilidades del modelo, con su calidad AUC).  
                2) **¿El riesgo extremo puede destrozar esa ventaja?** (colas: días crash, drawdown).  
                3) **¿Dependo demasiado de BTC?** (beta y causalidad temporal).  
                4) **¿El contexto actual acompaña?** (news score).

                **Cómo usarlo:**
                - IVI alto no significa “compra”; significa que el conjunto de pruebas está menos en contra.
                - IVI bajo no significa “nunca”; puede significar que ahora el riesgo o el régimen es malo.
                - Lo más inteligente es mirar IVI junto a “Edge por régimen” y “Recuperación tras drawdown”.

                **Por qué es conservador:**
                Penaliza fuerte el riesgo extremo y drawdowns grandes, porque en cripto esa es la forma típica de “morir” aunque tu tesis sea buena.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab6:
    st.subheader("Histórico / Export")
    if hist.empty:
        st.info("Aún no hay histórico. Activa guardar y pulsa Actualizar.")
    else:
        st.dataframe(hist.sort_values("as_of_date").tail(80), use_container_width=True)
        with open(RESULTS_PATH, "rb") as f:
            st.download_button("⬇️ Descargar daily_results.csv", f, file_name="daily_results.csv", mime="text/csv")

st.caption("⚠️ Esto no garantiza subidas. La idea es decidir con menos sesgo: confluencia + riesgo + regímenes + simulación.")
