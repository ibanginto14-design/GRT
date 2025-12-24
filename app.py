import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

import ccxt

from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from arch import arch_model
import ruptures as rpt


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="CryptoStatLab", page_icon="📈", layout="wide")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(DATA_DIR, "daily_results.csv")

FALLBACK_EXCHANGES = ["kraken", "coinbase", "bitstamp"]


# -----------------------------
# HELPERS
# -----------------------------
def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _symbol_variants(symbol: str):
    base, quote = symbol.split("/")
    variants = [symbol]
    if quote.upper() == "USDT":
        variants += [f"{base}/USD", f"{base}/USDC"]
    return variants


@st.cache_data(ttl=60 * 60)  # cache 1h
def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int = 365):
    """
    Descarga OHLCV con fallback automático si un exchange está bloqueado (p.ej. Binance 451).
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

            # Si es bloqueo 451 / restricted location, probamos siguiente exchange
            if ("451" in msg) or ("restricted location" in msg) or ("eligibility" in msg):
                continue

            # Otros errores: seguimos probando también
            continue

    raise RuntimeError(f"No se pudo descargar OHLCV. Último error: {last_err}")


def compute_returns(df: pd.DataFrame):
    df = df.copy()
    df["log_close"] = np.log(df["close"])
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = df["log_close"].diff()
    return df


def trend_regression(df: pd.DataFrame, window: int = 90):
    d = df.dropna().tail(window).copy()
    y = d["log_close"].values
    x = np.arange(len(d))
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return {"trend_slope": float(model.params[1]), "trend_pvalue": float(model.pvalues[1]), "trend_r2": float(model.rsquared)}


def stationarity_tests(df: pd.DataFrame):
    d = df["log_ret"].dropna()
    adf = adfuller(d, autolag="AIC")
    kpss_stat, kpss_p, _, _ = kpss(d, regression="c", nlags="auto")
    return {
        "adf_stat": float(adf[0]),
        "adf_pvalue": float(adf[1]),
        "kpss_stat": float(kpss_stat),
        "kpss_pvalue": float(kpss_p),
    }


def momentum_metrics(df: pd.DataFrame):
    d = df.dropna().copy()
    r = d["ret"].dropna()

    r30 = float((1 + r.tail(30)).prod() - 1) if len(r) >= 30 else np.nan
    r90 = float((1 + r.tail(90)).prod() - 1) if len(r) >= 90 else np.nan

    recent = r.tail(14)
    z = float((recent.mean() - r.mean()) / (r.std() + 1e-12)) if len(r) > 30 else np.nan
    return {"mom_ret_30d": r30, "mom_ret_90d": r90, "mom_z_14d": z}


def volume_signal(df: pd.DataFrame):
    d = df.dropna().copy()
    v = d["volume"]
    hist = v.tail(180)
    recent = v.tail(14).mean()
    z = float((recent - hist.mean()) / (hist.std() + 1e-12)) if len(hist) > 30 else np.nan
    return {"vol_z_14d": z}


def garch_volatility(df: pd.DataFrame):
    r = df["ret"].dropna() * 100.0
    if len(r) < 200:
        return {"garch_vol_now": np.nan}
    am = arch_model(r, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
    res = am.fit(disp="off")
    return {"garch_vol_now": float(res.conditional_volatility.iloc[-1])}


def structural_breaks(df: pd.DataFrame):
    d = df.dropna().copy()
    y = d["log_close"].values
    if len(y) < 120:
        return {"breakpoints_n": np.nan}
    algo = rpt.Pelt(model="rbf").fit(y)
    bkps = algo.predict(pen=8)
    n = max(0, len(bkps) - 1)
    return {"breakpoints_n": float(n)}


def rolling_correlation(df_asset: pd.DataFrame, df_bench: pd.DataFrame, window: int = 60):
    a = df_asset["ret"].rename("asset")
    b = df_bench["ret"].rename("bench")
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < window + 5:
        return {"corr_60d": np.nan}
    corr = joined["asset"].rolling(window).corr(joined["bench"]).iloc[-1]
    return {"corr_60d": float(corr)}


def simple_backtest_prob(df: pd.DataFrame):
    d = df.dropna().copy()
    d["ret_fwd_14"] = d["close"].shift(-14) / d["close"] - 1

    d["mom_30"] = (1 + d["ret"]).rolling(30).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    vol_hist = d["volume"].rolling(180)
    d["vol_z_14"] = (d["volume"].rolling(14).mean() - vol_hist.mean()) / (vol_hist.std() + 1e-12)

    sub = d.dropna(subset=["mom_30", "vol_z_14", "ret_fwd_14"]).copy()
    if len(sub) < 200:
        return {"bt_p_up_14d": np.nan, "bt_n": np.nan}

    cond = (sub["vol_z_14"] > 0) & (sub["mom_30"] > 0)
    hits = (sub.loc[cond, "ret_fwd_14"] > 0).mean() if cond.sum() > 20 else np.nan
    return {"bt_p_up_14d": float(hits) if hits == hits else np.nan, "bt_n": float(cond.sum())}


def scoreboard(metrics: dict):
    score = 50.0

    slope = metrics.get("trend_slope", np.nan)
    pval = metrics.get("trend_pvalue", np.nan)
    if slope == slope and pval == pval:
        if slope > 0 and pval < 0.05:
            score += 15
        elif slope > 0:
            score += 7
        elif slope < 0 and pval < 0.05:
            score -= 15
        elif slope < 0:
            score -= 7

    m30 = metrics.get("mom_ret_30d", np.nan)
    if m30 == m30:
        score += 10 if m30 > 0 else -10

    vz = metrics.get("vol_z_14d", np.nan)
    if vz == vz:
        score += 8 if vz > 0.5 else (-8 if vz < -0.5 else 0)

    gv = metrics.get("garch_vol_now", np.nan)
    if gv == gv:
        score += 3 if gv < 4 else (-6 if gv > 8 else -2)

    corr = metrics.get("corr_60d", np.nan)
    if corr == corr:
        score += 4 if corr < 0.5 else 0

    bp = metrics.get("breakpoints_n", np.nan)
    if bp == bp:
        score -= min(10, bp * 2)

    p_up = metrics.get("bt_p_up_14d", np.nan)
    if p_up == p_up:
        score += 10 if p_up > 0.6 else (-10 if p_up < 0.45 else 0)

    return float(np.clip(score, 0, 100))


def append_results(row: dict, path: str = RESULTS_PATH):
    df = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        out = pd.concat([old, df], ignore_index=True)
        out = out.drop_duplicates(subset=["as_of_date", "exchange", "symbol"], keep="last")
    else:
        out = df
    out.to_csv(path, index=False)


def load_results(path: str = RESULTS_PATH):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


# -----------------------------
# UI
# -----------------------------
st.title("📈 CryptoStatLab — Panel de pruebas estadísticas (actualizable diario)")
st.caption("No es una predicción, es un panel de **indicadores probabilísticos** (confluencia).")

colA, colB, colC, colD = st.columns([1.2, 1.2, 1, 1])
with colA:
    exchange_id = st.selectbox("Exchange (preferido)", ["binance", "kraken", "coinbase", "bitstamp"], index=0)
with colB:
    symbol = st.text_input("Símbolo (ej: BTC/USDT, ETH/USDT)", value="BTC/USDT")
with colC:
    limit = st.number_input("Días (OHLCV)", min_value=200, max_value=2000, value=600, step=50)
with colD:
    bench_symbol = st.text_input("Benchmark (para correlación)", value="BTC/USDT")

timeframe = "1d"

st.divider()
c1, c2 = st.columns([1, 2])

with c1:
    do_update = st.button("🔄 Actualizar ahora (calcular y guardar)", type="primary")
    st.info("Si Binance bloquea (451), se usará automáticamente Kraken/Coinbase/Bitstamp.")
    st.write(f"🕒 Hora actual: **{now_utc_iso()}**")

with c2:
    st.subheader("Histórico de resultados guardados")
    res_df = load_results()
    if res_df.empty:
        st.write("Aún no hay resultados guardados.")
    else:
        st.dataframe(res_df.sort_values("as_of_date").tail(30), use_container_width=True)

st.divider()

# -----------------------------
# UPDATE (GUARDAR)
# -----------------------------
if do_update:
    try:
        df = fetch_ohlcv(exchange_id, symbol, timeframe, limit=int(limit))
        df = compute_returns(df)

        dfb = fetch_ohlcv(exchange_id, bench_symbol, timeframe, limit=int(limit))
        dfb = compute_returns(dfb)

        metrics = {}
        metrics.update(trend_regression(df, window=90))
        metrics.update(stationarity_tests(df))
        metrics.update(momentum_metrics(df))
        metrics.update(volume_signal(df))
        metrics.update(garch_volatility(df))
        metrics.update(structural_breaks(df))
        metrics.update(rolling_correlation(df, dfb, window=60))
        metrics.update(simple_backtest_prob(df))

        score = scoreboard(metrics)

        as_of_date = str(df.index.max())
        row = {
            "as_of_date": as_of_date,
            "exchange": df.attrs.get("exchange_used", exchange_id),
            "symbol": df.attrs.get("symbol_used", symbol),
            "benchmark": dfb.attrs.get("symbol_used", bench_symbol),
            "updated_at_utc": now_utc_iso(),
            "score_0_100": score,
            **{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in metrics.items()},
        }
        append_results(row)

        st.success(
            f"✅ Guardado {row['symbol']} (exchange: {row['exchange']}) — fecha datos: {as_of_date} — Score: {score:.1f}/100"
        )

    except Exception as e:
        st.error(f"Error al actualizar: {e}")

# -----------------------------
# LIVE VIEW
# -----------------------------
st.subheader("Vista actual (sin guardar)")

try:
    df_live = fetch_ohlcv(exchange_id, symbol, timeframe, limit=int(limit))
    df_live = compute_returns(df_live)

    dfb_live = fetch_ohlcv(exchange_id, bench_symbol, timeframe, limit=int(limit))
    dfb_live = compute_returns(dfb_live)

    st.caption(
        f"Datos usados: {df_live.attrs.get('exchange_used', exchange_id)} — {df_live.attrs.get('symbol_used', symbol)}"
    )

    metrics_live = {}
    metrics_live.update(trend_regression(df_live, window=90))
    metrics_live.update(stationarity_tests(df_live))
    metrics_live.update(momentum_metrics(df_live))
    metrics_live.update(volume_signal(df_live))
    metrics_live.update(garch_volatility(df_live))
    metrics_live.update(structural_breaks(df_live))
    metrics_live.update(rolling_correlation(df_live, dfb_live, window=60))
    metrics_live.update(simple_backtest_prob(df_live))

    score_live = scoreboard(metrics_live)

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Score (0-100)", f"{score_live:.1f}")
    top2.metric("Trend slope (90d)", f"{metrics_live['trend_slope']:.4f}")
    top3.metric("Momentum 30d", f"{metrics_live['mom_ret_30d']*100:.2f}%")
    top4.metric("Vol z (14d)", f"{metrics_live['vol_z_14d']:.2f}")

    st.dataframe(pd.DataFrame(metrics_live, index=["value"]).T, use_container_width=True)

    st.subheader("Precio y volumen")
    st.line_chart(df_live["close"].astype(float))
    st.line_chart(df_live["volume"].astype(float))

except Exception as e:
    st.warning(f"No se pudo cargar vista actual: {e}")

st.divider()
st.subheader("Exportar histórico")
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "rb") as f:
        st.download_button("⬇️ Descargar daily_results.csv", f, file_name="daily_results.csv", mime="text/csv")
else:
    st.caption("Cuando actualices al menos una vez, aparecerá el export.")
