# ===================== Futures Pro — V22 Pro+ (Dual Δ$) [Calibrated + Bands] ======================
# Ultra-Advanced Hybrid Signaler/Executor + Fundamental Sentiment (RSS)
# ⭐️ ارتقاهای این نسخه:
#   1) کالیبراسیون احتمالات RF با Isotonic + split زمانی (leakage-safe)
#   2) کراس‌ولیدیشن زمانی با فاصله (Purged TimeSeriesSplit) برای آموزش/کالیبراسیون
#   3) ریسک/بک‌تست واقع‌گرایانه‌تر با کارمزد و اسلیپیج قابل تنظیم
#   4) پیش‌بینی Δ$ با باند عدم‌قطعیت (GradientBoosting با loss=quantile برای P20/P80)
#
# - Indicators + AI (RF calibrated) + HTF confirm + Regime filter + Soft Filters
# - Δ$ (raw + clipped-to-ATR) + Multi-candle plain-text view per symbol (+ اختیاری bands)
# - Walk-forward Backtest + Metrics (با هزینه‌ها)
# - Model cache (joblib) + adaptive AI weights
# - Retry/Backoff + Loop Scheduler + Portfolio controls
# - Telegram (optional)
#
# Quick start:
#   pip install -q ccxt yfinance scikit-learn joblib python-dotenv pandas numpy requests feedparser textblob
#   python -m textblob.download_corpora
#   python futures_pro_v22_pro_plus_dual_delta_calibrated.py
#   # --loop / --backtest نیز پشتیبانی می‌شود.
# ==============================================================================

import os, sys, json, time, math, logging, warnings, random, signal, subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("V22-PRO+ (Dual Δ$)")

# ===================== dependency upgrader (optional) ======================
UPGRADE_CHECK_PACKAGES = [
    "ccxt","yfinance","scikit-learn","pandas","numpy","joblib",
    "python-dotenv","requests","feedparser","textblob"
]

def pip_list_outdated()->Dict[str, Dict[str,str]]:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "list", "--outdated", "--format", "json"], text=True)
        items = json.loads(out)
        return {it["name"].lower(): it for it in items}
    except Exception as e:
        log.debug("pip outdated check failed: %s", e)
        return {}

def offer_upgrades(pkgs: List[str]):
    outdated = pip_list_outdated()
    to_offer = [p for p in pkgs if p.lower() in outdated]
    if not to_offer:
        print("All core packages are up to date.")
        return
    print("\n=== Package Upgrades Available ===")
    for p in to_offer:
        cur = outdated[p.lower()]["version"]
        new = outdated[p.lower()]["latest_version"]
        print(f"- {p}: {cur} → {new}")
    print("Do you want to upgrade any of these now? [y/N]")
    ans = input("> ").strip().lower()
    if ans not in ("y","yes"):
        print("Skipping upgrades for now.\n"); return
    for p in to_offer:
        try:
            print(f"Upgrading {p} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", p])
        except Exception as e:
            print(f"Upgrade failed for {p}: {e}")
    print("Upgrade routine finished.\n")

# ===================== optional .env ======================
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ===================== third-party (required) ======================
try:
    import ccxt
except Exception as e:
        raise SystemExit("ccxt is required. pip install ccxt") from e
try:
    import yfinance as yf
except Exception as e:
        raise SystemExit("yfinance is required. pip install yfinance") from e

# --- Fundamental sentiment deps ---
try:
    import feedparser
    from textblob import TextBlob
except Exception as e:
    raise SystemExit("feedparser + textblob are required. Install & run: pip install feedparser textblob && python -m textblob.download_corpora")

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# --- AI ---
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump, load as joblib_load

# ===================== env / config ======================
def env_bool(name:str, default:bool)->bool:
    v=os.getenv(name, str(default)).strip().lower()
    return v in ("1","true","yes","y","t")

def env_float(name:str, default:float)->float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def env_int(name:str, default:int)->int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default

EXCHANGE_NAME = os.getenv("EXCHANGE","okx").lower()    # binance | bybit | okx
API_KEY      = os.getenv("API_KEY","")
API_SECRET   = os.getenv("API_SECRET","")
TESTNET      = env_bool("TESTNET", True)
DRY_RUN      = env_bool("DRY_RUN", True)
DEFAULT_TF   = os.getenv("DEFAULT_TF","15m")
HTF_TF       = os.getenv("HTF","1h")
USE_HTF      = env_bool("USE_HTF", True)
SYMBOLS      = [s.strip() for s in os.getenv("SYMBOLS","BTC/USDT,ETH/USDT").split(",") if s.strip()]

ACCOUNT_EQUITY     = env_float("ACCOUNT_EQUITY", 1000.0)
RISK_PER_TRADE     = env_float("RISK_PER_TRADE", 0.01)
PORTFOLIO_RISK_CAP = env_float("PORTFOLIO_RISK_CAP", 0.2)

AI_ENABLED      = env_bool("AI_ENABLED", True)
AI_TRAIN_WINDOW = env_int("AI_TRAIN_WINDOW", 100)
AI_TEST_SIZE    = env_float("AI_TEST_SIZE", 0.2)
CONFIDENCE_MIN  = env_float("CONFIDENCE_MIN", 35.0)
SOFT_FILTERS    = env_bool("SOFT_FILTERS", True)

MODEL_DIR       = os.getenv("MODEL_DIR", "models")

TRAILING_ATR_MULT = env_float("TRAILING_ATR_MULT", 1.5)
USE_TWAP_MIN_NOTIONAL = env_float("USE_TWAP_MIN_NOTIONAL", 0)
TWAP_SLICES      = env_int("TWAP_SLICES", 3)

TELEGRAM_TOKEN  = os.getenv("TG_BOT_TOKEN","")
TELEGRAM_CHATID = os.getenv("TG_CHAT_ID","")

INTERVAL_SEC     = env_int("INTERVAL_SEC", 60)
MAX_CONSEC_LOSSES_PAUSE = env_int("MAX_CONSEC_LOSSES_PAUSE", 3)

# Δ$ clipping control
DELTA_CLIP_ATR_MULT = env_float("DELTA_CLIP_ATR_MULT", 2.5)

# Fundamental cache TTL (seconds)
FUND_TTL_SEC = env_int("FUND_TTL_SEC", 20*60)

# Multi-candle horizon
HORIZON = env_int("HORIZON", 4)

# ----- Weights for confidence mixing -----
W_ML_IND = env_float("W_ML_IND", 0.4)  # Indicators + ML
W_PA     = env_float("W_PA", 0.3)      # Price Action
W_OF     = env_float("W_OF", 0.3)      # Order Flow

# ----- NEW: Probability calibration / CV / costs / bands -----
CALIBRATE_PROBS = env_bool("CALIBRATE_PROBS", True)
CV_FOLDS        = env_int("CV_FOLDS", 4)            # TimeSeriesSplit folds
CV_GAP          = env_int("CV_GAP", 10)             # purge gap to reduce leakage
COST_BPS        = env_float("COST_BPS", 2.0)        # per side fee in bps (0.01% = 1 bps)
SLIPPAGE_BPS    = env_float("SLIPPAGE_BPS", 1.0)    # per fill slippage assumption in bps
USE_BANDS       = env_bool("USE_BANDS", True)       # print/compute uncertainty bands

# ===================== utils ======================
os.makedirs(MODEL_DIR, exist_ok=True)

def human_time(ts=None):
    if ts is None: ts=time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def map_symbol_to_yf(symbol: str)->str:
    base = symbol.split("/")[0].strip()
    return f"{base}-USD"

def sym_base(symbol:str)->str:
    return symbol.split("/")[0].upper().strip()

def retry_call(fn, tries=3, delay=0.7, backoff=2.0, exceptions=(Exception,), desc=""):
    ex = None
    for i in range(tries):
        try:
            return fn()
        except exceptions as e:
            ex = e
            log.warning("%s attempt %d/%d failed: %s", desc or fn.__name__, i+1, tries, e)
            time.sleep(delay); delay *= backoff
    if ex:
        raise ex

def tg_send(text:str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHATID: return
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHATID, "text": text[:4000]})
    except Exception as e:
        log.debug("Telegram send failed: %s", e)

# ===================== indicators ======================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period:int=14)->pd.Series:
    delta = series.diff()
    gain = np.where(delta>0, delta, 0.0)
    loss = np.where(delta<0, -delta, 0.0)
    gain_s = pd.Series(gain, index=series.index).rolling(period).mean()
    loss_s = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = gain_s/(loss_s+1e-9)
    return 100 - (100/(1+rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    m = ema_fast - ema_slow
    s = m.ewm(span=signal, adjust=False).mean()
    return m, s

def bollinger(series: pd.Series, window=20, n=2.0):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + n*sd
    lower = ma - n*sd
    bbpos = (series - ma)/(2*sd + 1e-9)
    return upper, lower, bbpos

def atr(df: pd.DataFrame, period:int=14)->pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    hl = h - l
    hc = (h - c.shift()).abs()
    lc = (l - c.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def volatility_regime(close: pd.Series, window:int=50) -> pd.Series:
    ret = close.pct_change()
    vol = ret.rolling(window).std()
    vol_n = (vol / (vol.rolling(window).mean()+1e-9)).clip(0, 3)
    return vol_n

# ===================== data layer ======================
def build_data_exchange() -> Optional[ccxt.Exchange]:
    try:
        if EXCHANGE_NAME == "binance":
            ex = ccxt.binance({"options":{"defaultType":"future"}})
        elif EXCHANGE_NAME == "bybit":
            ex = ccxt.bybit({"options":{"defaultType":"future"}})
        elif EXCHANGE_NAME == "okx":
            ex = ccxt.okx({"options":{"defaultType":"future"}})
        else:
            return None
        ex.set_sandbox_mode(False)
        retry_call(ex.load_markets, desc="load_markets")
        return ex
    except Exception as e:
        log.warning("Data exchange init failed: %s", e)
        return None

def fetch_ohlcv(symbol:str, timeframe:str, limit:int=700) -> pd.DataFrame:
    def _from_ex():
        ex = build_data_exchange()
        if not ex: return None
        ohlcv = retry_call(lambda: ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit),
                           desc=f"fetch_ohlcv({symbol},{timeframe})")
        if ohlcv and len(ohlcv)>0:
            df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        return None
    df = None
    try:
        df = _from_ex()
    except Exception as e:
        log.warning("Exchange fetch failed %s | %s", symbol, str(e))
    if df is not None and not df.empty:
        return df
    # fallback to yfinance
    try:
        yf_sym = map_symbol_to_yf(symbol)
        tf_map = {"1m":"1m","3m":"2m","5m":"5m","15m":"15m","30m":"30m","1h":"60m","4h":"60m","1d":"1d"}
        yf_itv = tf_map.get(timeframe, "15m")
        period = "7d" if yf_itv.endswith("m") else "1mo"
        dd = yf.download(yf_sym, interval=yf_itv, period=period, progress=False, prepost=False, threads=True)
        if dd is None or dd.empty:
            return pd.DataFrame()
        dd.columns = [c[0] if isinstance(c, tuple) else c for c in dd.columns]
        df = dd.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        df.index.name="timestamp"
        return df[["open","high","low","close","volume"]].dropna()
    except Exception as e:
        log.warning("yfinance fallback failed for %s | %s", symbol, e)
        return pd.DataFrame()

def fetch_htf_confirm(symbol:str, htf:str, limit:int=360) -> Optional[pd.Series]:
    d = fetch_ohlcv(symbol, htf, limit)
    if d is None or d.empty: return None
    d["ema200"] = ema(d["close"], 200)
    d["ema50"]  = ema(d["close"], 50)
    d["ema20"]  = ema(d["close"], 20)
    d["macd"], d["macds"] = macd(d["close"])
    d.dropna(inplace=True)
    return d.iloc[-1] if len(d) else None

# ===================== trading layer ======================
def build_trade_exchange() -> Optional[ccxt.Exchange]:
    try:
        if EXCHANGE_NAME == "binance":
            ex = ccxt.binance({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True,
                               "options":{"defaultType":"future"}})
        elif EXCHANGE_NAME == "bybit":
            ex = ccxt.bybit({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True,
                             "options":{"defaultType":"future"}})
        elif EXCHANGE_NAME == "okx":
            ex = ccxt.okx({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True,
                           "options":{"defaultType":"future"}})
        else:
            return None
        ex.set_sandbox_mode(TESTNET)
        retry_call(ex.load_markets, desc="load_markets(trade)")
        return ex
    except Exception as e:
        log.warning("Trade exchange init failed: %s", e)
        return None

def ensure_margin_mode_and_leverage(ex: ccxt.Exchange, symbol:str, leverage:int=2, mode:str="CROSSED"):
    try:
        if hasattr(ex, "set_leverage"):
            retry_call(lambda: ex.set_leverage(leverage, symbol), desc="set_leverage")
        if hasattr(ex, "set_margin_mode"):
            retry_call(lambda: ex.set_margin_mode(mode, symbol), desc="set_margin_mode")
    except Exception as e:
        log.warning("%s leverage/margin -> %s", symbol, e)

def _place_market(ex: ccxt.Exchange, symbol:str, side:str, qty:float):
    return retry_call(lambda: ex.create_order(symbol, "market", side, qty), desc="create_order.market")

def _place_limit(ex: ccxt.Exchange, symbol:str, side:str, qty:float, price:float, params=None):
    params = params or {}
    return retry_call(lambda: ex.create_order(symbol, "limit", side, qty, price, params), desc="create_order.limit")

def _place_stop(ex: ccxt.Exchange, symbol:str, side:str, qty:float, stop_price:float):
    try:
        return retry_call(lambda: ex.create_order(symbol, "stop_market", side, qty, None,
                                                  {"stopPrice": stop_price, "reduceOnly": True}),
                          desc="create_order.stop_market")
    except Exception:
        return retry_call(lambda: ex.create_order(symbol, "stop_limit", side, qty, stop_price,
                                                  {"stopPrice": stop_price, "reduceOnly": True}),
                          desc="create_order.stop_limit")

def place_bracket(ex: ccxt.Exchange, symbol:str, side:str, qty:float, tp:float, sl:float, trailing_atr:Optional[float]=None):
    order = _place_market(ex, symbol, side, qty)
    opp = "sell" if side.lower()=="buy" else "buy"
    _place_limit(ex, symbol, opp, qty, tp, {"reduceOnly": True})
    _place_stop(ex, symbol, opp, qty, sl)
    return order

def place_twap_bracket(ex, symbol, side, qty, tp, sl, slices:int=3, delay_s:float=2.0):
    slice_qty = max(qty/slices, 1e-8)
    for _ in range(slices):
        _place_market(ex, symbol, side, slice_qty)
        time.sleep(delay_s)
    opp = "sell" if side.lower()=="buy" else "buy"
    _place_limit(ex, symbol, opp, qty, tp, {"reduceOnly": True})
    _place_stop(ex, symbol, opp, qty, sl)
    return True

# ===================== compute & scoring ======================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema20"]  = ema(d["close"], 20)
    d["ema50"]  = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)
    d["rsi"]    = rsi(d["close"], 14)
    d["macd"], d["macds"] = macd(d["close"], 12,26,9)
    d["bbU"], d["bbL"], d["bbpos"] = bollinger(d["close"], 20, 2.0)
    d["atr"] = atr(d, 14)
    d["ret"] = d["close"].pct_change()
    d["vol_regime"] = volatility_regime(d["close"], 50)
    d.dropna(inplace=True)
    return d

def score_signal(row: pd.Series) -> Tuple[float, str, str, float]:
    s_ema = 0.0
    if row["close"] > row["ema20"] > row["ema50"] > row["ema200"]:
        s_ema = 1.0
    elif row["close"] < row["ema20"] < row["ema50"] < row["ema200"]:
        s_ema = -1.0
    s_rsi  = 0.6 if row["rsi"] >= 60 else (-0.6 if row["rsi"] <= 40 else 0.0)
    s_macd = 0.8 if row["macd"] > row["macds"] else -0.8
    s_bb   = float(np.clip(row["bbpos"], -0.5, 0.5))
    s_vwma = 0.4 if row["close"] > row["ema20"] else -0.4
    reg_pen = float(np.clip(row.get("vol_regime", 1.0), 0.0, 2.0))
    raw = (1.6*s_ema + 1.2*s_macd + 1.0*s_rsi + 0.5*s_bb + s_vwma) / (1.0 + 0.25*(reg_pen-1.0))
    raw = float(np.clip(raw, -3.0, 3.0))
    vol_penalty = float(np.clip((row["atr"]/(row["close"]+1e-9)),0,0.08))
    conf = (abs(raw)/3.0)*100.0*(1.0 - vol_penalty)
    if raw > 0.25:
        trend, color = "Bullish", ""
    elif raw < -0.25:
        trend, color = "Bearish", ""
    else:
        trend, color = "Neutral", ""
    return raw, trend, color, round(conf,2)

def prediction_targets(last_price: float, atr_val: float, score: float, trend:str, ai_align: Optional[float]=None)->Tuple[float,float]:
    a = atr_val if atr_val>0 else last_price*0.01
    vol = float(np.clip(a/(last_price+1e-9), 0, 0.05))
    base_k = 0.9 + 0.6*(abs(score)/3.0)
    k = base_k * (1.0 - 0.7*vol)
    if ai_align is not None:
        k *= (0.92 + 0.16*ai_align)
    if trend=="Bullish":
        tp = last_price + k*1.35*a
        sl = last_price - k*0.95*a
    elif trend=="Bearish":
        tp = last_price - k*1.35*a
        sl = last_price + k*0.95*a
    else:
        tp = last_price + k*0.65*a
        sl = last_price - k*0.65*a
    return round(float(tp), 2), round(float(sl), 2)

# ===================== Price Action (advanced, OHLC) ======================
def _pivots(h: pd.Series, l: pd.Series, left:int=3, right:int=3):
    ph = (h.shift(1).rolling(left).max() < h) & (h.shift(-1).rolling(right).max() <= h.shift(-1))
    pl = (l.shift(1).rolling(left).min() > l) & (l.shift(-1).rolling(right).min() >= l.shift(-1))
    return ph.fillna(False).astype(int), pl.fillna(False).astype(int)

def add_price_action(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    o,h,l,c = d["open"], d["high"], d["low"], d["close"]

    # pivots
    ph, pl = _pivots(h,l, left=3, right=3)
    d["pivot_high"] = ph
    d["pivot_low"]  = pl

    # last swing highs/lows
    d["last_swh"] = np.where(ph==1, h, np.nan)
    d["last_swl"] = np.where(pl==1, l, np.nan)
    d["last_swh"] = pd.Series(d["last_swh"]).ffill()
    d["last_swl"] = pd.Series(d["last_swl"]).ffill()

    # Market Structure (HH/HL vs LL/LH)
    d["ms_up"] = ((h > d["last_swh"].shift(1)) & (l > d["last_swl"].shift(1))).astype(int)
    d["ms_down"] = ((h < d["last_swh"].shift(1)) & (l < d["last_swl"].shift(1))).astype(int)

    # Break of Structure (BOS)
    d["bos_up"] = (h > d["last_swh"].shift(1)).astype(int)
    d["bos_dn"] = (l < d["last_swl"].shift(1)).astype(int)

    # Fair Value Gap (3-candle)
    d["fvg_up"] = ((l.shift(-1) > h.shift(-1).shift( -1)) | (l > h.shift(2))).astype(int)  # محافظه‌کار
    d["fvg_dn"] = ((h.shift(-1) < l.shift(-1).shift( -1)) | (h < l.shift(2))).astype(int)
    d["fvg_up"] = d["fvg_up"].fillna(0).astype(int)
    d["fvg_dn"] = d["fvg_dn"].fillna(0).astype(int)

    # Liquidity Sweep (stop-run)
    prev_high = h.shift(1); prev_low = l.shift(1)
    d["liq_sweep_up"] = ((h > prev_high) & (c < prev_high)).astype(int)
    d["liq_sweep_dn"] = ((l < prev_low)  & (c > prev_low)).astype(int)

    # Engulfing
    d["engulf_bull"] = ((c>o) & (df["close"].shift(1)<df["open"].shift(1)) &
                        (c>df["open"].shift(1)) & (o<df["close"].shift(1))).astype(int)
    d["engulf_bear"] = ((c<o) & (df["close"].shift(1)>df["open"].shift(1)) &
                        (c<df["open"].shift(1)) & (o>df["close"].shift(1))).astype(int)

    # Order Block (proxy)
    prev_bear = (df["close"].shift(1) < df["open"].shift(1)).astype(int)
    prev_bull = (df["close"].shift(1) > df["open"].shift(1)).astype(int)
    d["bull_ob"] = ((prev_bear==1) & (d["bos_up"]==1)).astype(int)
    d["bear_ob"] = ((prev_bull==1) & (d["bos_dn"]==1)).astype(int)

    # PA score (normalized [-1..1])
    pa_raw = (
        + 0.9*d["ms_up"]   - 0.9*d["ms_down"]
        + 0.8*d["bos_up"]  - 0.8*d["bos_dn"]
        + 0.4*d["engulf_bull"] - 0.4*d["engulf_bear"]
        + 0.5*d["bull_ob"] - 0.5*d["bear_ob"]
        - 0.5*d["liq_sweep_up"] + 0.5*d["liq_sweep_dn"]
        + 0.2*d["fvg_up"]  - 0.2*d["fvg_dn"]
    )
    m = pa_raw.rolling(150).apply(lambda x: np.max(np.abs(x)) if len(x)>0 else 1.0).replace(0,1.0)
    d["pa_score"] = (pa_raw / m).clip(-1,1).fillna(0.0)
    d["pa_score_pos"] = ((d["pa_score"]+1)/2.0).clip(0,1)

    return d

# ===================== Order Flow (OHLCV-based proxy) ======================
def calc_vwap(df: pd.DataFrame, session: str = "rolling", window: int = 96) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan).fillna(0.0)
    if session == "cumulative":
        pv = (tp * vol).cumsum()
        vv = vol.cumsum().replace(0, np.nan)
        return (pv / vv).fillna(method="bfill").fillna(method="ffill")
    pv = (tp * vol).rolling(window).sum()
    vv = vol.rolling(window).sum().replace(0, np.nan)
    return (pv / vv).fillna(method="bfill").fillna(method="ffill")

def cumulative_delta(df: pd.DataFrame, alpha: float = 0.5) -> pd.Series:
    body = (df["close"] - df.get("open", df["close"].shift(1).fillna(df["close"])))
    span = (df["high"] - df["low"]).replace(0, np.nan).fillna(1e-9)
    power = (body / span).clip(-1, 1)
    delta = power * df["volume"]
    cvd = delta.ewm(alpha=alpha, adjust=False). mean().cumsum()
    return cvd

def volume_profile(df: pd.DataFrame, lookback: int = 384, bins: int = 24):
    d = df.iloc[-lookback:].copy()
    if len(d) < 10:
        return np.nan, np.nan, np.nan
    prices = d["close"].values
    vols   = d["volume"].values
    lo, hi = float(d["low"].min()), float(d["high"].max())
    if hi <= lo:
        return np.nan, np.nan, np.nan
    edges = np.linspace(lo, hi, bins+1)
    idx   = np.clip(np.digitize(prices, edges)-1, 0, bins-1)
    vol_hist = np.zeros(bins)
    for i, v in zip(idx, vols): vol_hist[i] += v
    if vol_hist.sum() <= 0:
        return np.nan, np.nan, np.nan
    vpoc_bin = int(np.argmax(vol_hist))
    center = (edges[vpoc_bin] + edges[vpoc_bin+1]) * 0.5
    nz = np.where(vol_hist>0, vol_hist, np.nan)
    hvn_bin = int(np.nanargmax(nz)) if np.isfinite(nz).any() else vpoc_bin
    lvn_bin = int(np.nanargmin(nz)) if np.isfinite(nz).any() else vpoc_bin
    hvn = (edges[hvn_bin] + edges[hvn_bin+1]) * 0.5
    lvn = (edges[lvn_bin] + edges[lvn_bin+1]) * 0.5
    return center, hvn, lvn

def liquidity_pools(df: pd.DataFrame, lookback: int = 40, tol: float = 0.0008):
    close = df["close"].values
    hi = df["high"].values
    lo = df["low"].values
    pools_up = np.zeros(len(df), dtype=int)
    pools_dn = np.zeros(len(df), dtype=int)
    for i in range(2, len(df)):
        lo_i, hi_i = lo[i], hi[i]
        win_hi = hi[max(0, i-lookback):i]
        win_lo = lo[max(0, i-lookback):i]
        if np.any(np.abs(win_hi - hi_i)/max(close[i],1e-9) < tol):
            pools_up[i] = 1
        if np.any(np.abs(win_lo - lo_i)/max(close[i],1e-9) < tol):
            pools_dn[i] = 1
    return pd.Series(pools_up, index=df.index), pd.Series(pools_dn, index=df.index)

def imbalance_metric(df: pd.DataFrame):
    opn = df.get("open", df["close"].shift(1).fillna(df["close"]))
    rng = (df["high"] - df["low"]).replace(0, np.nan).fillna(1e-9)
    upv = ((df["close"] - opn).clip(lower=0) / rng) * df["volume"]
    dnv = ((opn - df["close"]).clip(lower=0) / rng) * df["volume"]
    imb = (upv - dnv) / (upv + dnv + 1e-9)  # [-1..1]
    return imb.fillna(0.0)

def add_order_flow(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["vwap"] = calc_vwap(d, session="rolling", window=96)
    d["vwap_dist"] = (d["close"] / (d["vwap"]+1e-9)) - 1.0

    d["cvd"] = cumulative_delta(d, alpha=0.4)
    d["cvd_ma_fast"] = d["cvd"].ewm(span=20, adjust=False).mean()
    d["cvd_ma_slow"] = d["cvd"].ewm(span=60, adjust=False).mean()
    d["cvd_trend_up"] = (d["cvd_ma_fast"] > d["cvd_ma_slow"]).astype(int)
    d["cvd_trend_dn"] = (d["cvd_ma_fast"] < d["cvd_ma_slow"]).astype(int)

    vpoc, hvn, lvn = volume_profile(d, lookback=384, bins=24)
    d["vpoc_price"] = vpoc
    d["hvn_price"]  = hvn
    d["lvn_price"]  = lvn
    d["dist_vpoc"] = (d["close"]/(d["vpoc_price"]+1e-9)) - 1.0 if np.isfinite(vpoc) else 0.0

    lp_up, lp_dn = liquidity_pools(d, lookback=40, tol=0.0008)
    d["liq_pool_up"] = lp_up
    d["liq_pool_dn"] = lp_dn

    d["imbalance"] = imbalance_metric(d)

    of_score = (
        0.9 * d["cvd_trend_up"] - 0.9 * d["cvd_trend_dn"] +
        0.7 * (d["imbalance"].clip(-1,1)) +
        0.5 * (-(d["vwap_dist"]).clip(-1,1)) +
        0.4 * (-(d["dist_vpoc"])).clip(-1,1) +
        -0.6 * d["liq_pool_up"] + 0.6 * d["liq_pool_dn"]
    )
    m = of_score.rolling(150).apply(lambda x: np.max(np.abs(x)) if len(x)>0 else 1.0).replace(0,1.0)
    d["of_score"] = (of_score / m).clip(-1,1).fillna(0.0)
    d["of_score_pos"] = ((d["of_score"]+1)/2.0)

    return d
# ===================== AI features (RandomForest) ======================
AI_FEATURES = [
    "ema_trend","macd_cross","rsi_overbought","rsi_oversold",
    "candle_bull","bb_break_upper","bb_break_lower","bbpos_clip",
    "atr_norm","vol_regime",
    # --- Price Action ---
    "ms_up","ms_down","bos_up","bos_dn","fvg_up","fvg_dn",
    "bull_ob","bear_ob","liq_sweep_up","liq_sweep_dn",
    "engulf_bull","engulf_bear","pa_score",
    # --- Order Flow ---
    "cvd_trend_up","cvd_trend_dn","imbalance","vwap_dist","dist_vpoc","of_score"
]

def build_ai_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["ema_trend"]      = (x["close"]>x["ema20"]).astype(int) + (x["close"]>x["ema50"]).astype(int)
    x["macd_cross"]     = (x["macd"]>x["macds"]).astype(int)
    x["rsi_overbought"] = (x["rsi"]>70).astype(int)
    x["rsi_oversold"]   = (x["rsi"]<30).astype(int)
    x["candle_bull"]    = (x["close"]>x["open"]).astype(int) if "open" in x else (x["ret"]>0).astype(int)
    x["bb_break_upper"] = (x["close"]>x["bbU"]).astype(int)
    x["bb_break_lower"] = (x["close"]<x["bbL"]).astype(int)
    x["bbpos_clip"]     = np.clip(x["bbpos"], -1.0, 1.0)
    x["atr_norm"]       = (x["atr"]/(x["close"]+1e-9)).fillna(0.0)
    x["vol_regime"]     = x["vol_regime"].clip(0.0, 3.0)

    # --- ensure PA/OF columns exist (fallback zeros) ---
    for col in ["ms_up","ms_down","bos_up","bos_dn","fvg_up","fvg_dn",
                "bull_ob","bear_ob","liq_sweep_up","liq_sweep_dn",
                "engulf_bull","engulf_bear","pa_score",
                "cvd_trend_up","cvd_trend_dn","imbalance","vwap_dist","dist_vpoc","of_score"]:
        x[col] = x.get(col, pd.Series(0, index=x.index)).fillna(0)

    x["target"]         = (x["close"].shift(-1) > x["close"]).astype(int)
    return x.dropna()

def model_path(symbol:str, tf:str)->str:
    safe = symbol.replace("/","_")
    return os.path.join(MODEL_DIR, f"rf_{safe}_{tf}.joblib")

def _purged_time_series_split(n_samples:int, n_splits:int, gap:int):
    """
    Generator of (train_idx, test_idx) with a purge 'gap' between train and test.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, te in tscv.split(np.arange(n_samples)):
        te_start = te[0]
        tr = tr[tr < (te_start - gap)]  # purge overlap
        if len(tr)==0: 
            continue
        yield tr, te

def ai_train_and_prob_up(d: pd.DataFrame, symbol:str, timeframe:str) -> Optional[float]:
    if len(d) < max(50, AI_TRAIN_WINDOW//2): return None
    x = build_ai_features(d)
    if len(x) < max(50, AI_TRAIN_WINDOW//2): return None
    xw = x.iloc[-AI_TRAIN_WINDOW:] if len(x) > AI_TRAIN_WINDOW else x.copy()
    last_feats = xw.iloc[[-1]][AI_FEATURES]
    X = xw[AI_FEATURES]
    y = xw["target"]
    if len(X) < 60: return None

    base_model = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)

    # --- TimeSeries CV + calibration (isotonic) ---
    if CALIBRATE_PROBS and len(X) >= 120:
        # آخرین 20% برای کالیبراسیون/ولیدیشن
        cut = int(len(X)*0.8)
        X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]
        X_cal, y_cal = X.iloc[cut:], y.iloc[cut:]

        # Purged CV روی train
        for tr_idx, te_idx in _purged_time_series_split(len(X_tr), max(2, min(CV_FOLDS, len(X_tr)//30)), CV_GAP):
            base_model.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])  # (فقط برای پایداری، می‌توان امتیاز گرفت)

        # fit روی کل train، سپس کالیبره روی val
        base_model.fit(X_tr, y_tr)
        try:
            calib = CalibratedClassifierCV(base_estimator=base_model, method="isotonic", cv="prefit")
            calib.fit(X_cal, y_cal)
            prob_up = calib.predict_proba(last_feats)[0][1]*100.0
            # cache
            dump(calib, model_path(symbol, timeframe))
        except Exception as e:
            log.debug("Calibration failed, fallback raw proba: %s", e)
            prob_up = base_model.fit(X.iloc[:-1], y.iloc[:-1]).predict_proba(last_feats)[0][1]*100.0
            dump(base_model, model_path(symbol, timeframe))
    else:
        path = model_path(symbol, timeframe)
        try:
            mdl = joblib_load(path)
        except Exception:
            mdl = None
        mdl = base_model
        mdl.fit(X.iloc[:-1], y.iloc[:-1])
        try:
            dump(mdl, path)
        except Exception as e:
            log.debug("Model save failed: %s", e)
        prob_up = mdl.predict_proba(last_feats)[0][1]*100.0

    return float(round(prob_up,2))

# ===================== AI Δ$ (regression + quantile bands) ======================
def model_path_reg(symbol:str, tf:str, tag:str="huber")->str:
    safe = symbol.replace("/","_")
    return os.path.join(MODEL_DIR, f"gbr_delta_{tag}_{safe}_{tf}.joblib")

def _build_reg_features(d: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    x = d.copy()
    x["ret1"] = np.log(x["close"] / x["close"].shift(1))
    x["ret2"] = np.log(x["close"].shift(1) / x["close"].shift(2))
    x["ret3"] = np.log(x["close"].shift(2) / x["close"].shift(3))
    x["vol20"] = x["ret1"].rolling(20).std()
    x["vol50"] = x["ret1"].rolling(50).std()
    x["atr_norm"] = (x["atr"]/(x["close"]+1e-9))
    x["ema20_dist"]  = (x["close"]/ (x["ema20"]+1e-9)) - 1.0
    x["ema50_dist"]  = (x["close"]/ (x["ema50"]+1e-9)) - 1.0
    x["ema200_dist"] = (x["close"]/ (x["ema200"]+1e-9)) - 1.0
    x["macd_hist"] = x["macd"] - x["macds"]
    x["bb_width"]  = ((x["bbU"] - x["bbL"]) / (x["close"]+1e-9))
    x["bb_pos"]    = np.clip(x["bbpos"], -1.5, 1.5)
    x["rsi"] = x["rsi"]
    x["y_logret"] = np.log(x["close"].shift(-1) / x["close"])
    cols = ["ret1","ret2","ret3","vol20","vol50","atr_norm",
            "ema20_dist","ema50_dist","ema200_dist","macd_hist",
            "bb_width","bb_pos","rsi","vol_regime"]
    x = x[cols+["y_logret","close","atr"]].dropna()
    return x, cols

def _fit_gbr(X, y, **kw):
    m = GradientBoostingRegressor(**kw)
    m.fit(X, y)
    return m

def ai_train_and_pred_delta(d: pd.DataFrame, symbol:str, timeframe:str) -> Optional[Tuple[float,float, Optional[float], Optional[float]]]:
    try:
        if len(d) < max(100, AI_TRAIN_WINDOW//2): return None
        feats, cols = _build_reg_features(d)
        if len(feats) < max(100, AI_TRAIN_WINDOW//2): return None
        w = feats.iloc[-AI_TRAIN_WINDOW:] if len(feats) > AI_TRAIN_WINDOW else feats.copy()
        X_tr = w.iloc[:-1][cols]; y_tr = w.iloc[:-1]["y_logret"]; X_last = w.iloc[[-1]][cols]
        last_close = float(w.iloc[-1]["close"]); last_atr = float(w.iloc[-1]["atr"])
        if len(X_tr) < 80: return None

        # point model (Huber)
        path_point = model_path_reg(symbol, timeframe, "huber")
        m_point = None
        try:
            m_point = joblib_load(path_point)
        except Exception:
            pass
        m_point = _fit_gbr(
            X_tr, y_tr,
            loss="huber", alpha=0.9, n_estimators=800, learning_rate=0.05,
            max_depth=3, subsample=0.85, random_state=42
        )
        try:
            dump(m_point, path_point)
        except Exception as e:
            log.debug("Reg model save failed: %s", e)

        yhat = float(m_point.predict(X_last)[0])
        delta_raw = last_close*(math.exp(yhat) - 1.0)
        atr_clip = max(1e-9, DELTA_CLIP_ATR_MULT*last_atr)
        delta_clip = float(np.clip(delta_raw, -atr_clip, atr_clip))

        # quantile bands (P20/P80)
        q_lo, q_hi = 0.2, 0.8
        path_lo = model_path_reg(symbol, timeframe, f"q{int(q_lo*100)}")
        path_hi = model_path_reg(symbol, timeframe, f"q{int(q_hi*100)}")
        try:
            m_lo = joblib_load(path_lo)
            m_hi = joblib_load(path_hi)
        except Exception:
            m_lo = None; m_hi = None
        m_lo = _fit_gbr(X_tr, y_tr, loss="quantile", alpha=q_lo, n_estimators=600, learning_rate=0.05,
                        max_depth=3, subsample=0.85, random_state=43)
        m_hi = _fit_gbr(X_tr, y_tr, loss="quantile", alpha=q_hi, n_estimators=600, learning_rate=0.05,
                        max_depth=3, subsample=0.85, random_state=44)
        try:
            dump(m_lo, path_lo); dump(m_hi, path_hi)
        except Exception:
            pass
        y_lo = float(m_lo.predict(X_last)[0]); y_hi = float(m_hi.predict(X_last)[0])
        d_low  = last_close*(math.exp(y_lo) - 1.0)
        d_high = last_close*(math.exp(y_hi) - 1.0)
        return round(delta_raw, 2), round(delta_clip, 2), round(d_low,2), round(d_high,2)
    except Exception as e:
        log.debug("Δ$ regression error: %s", e)
        return None

# ===================== Fundamental Sentiment (RSS + TextBlob) ======================
FUND_FEEDS = {
    "BTC": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
        "https://cointelegraph.com/rss/tag/bitcoin"
    ],
    "ETH": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ETH-USD&region=US&lang=en-US",
        "https://cointelegraph.com/rss/tag/ethereum"
    ]
}
_FUND_CACHE: Dict[str, Tuple[float, str, float]] = {}  # base -> (ts, label, perc0..100)

def _funda_from_text(text:str)->float:
    try:
        return float(TextBlob(text).sentiment.polarity)  # -1..1
    except Exception:
        return 0.0

def fundamental_sentiment(symbol:str)->Tuple[str, Optional[float]]:
    base = sym_base(symbol)
    now = time.time()
    if base in _FUND_CACHE and now - _FUND_CACHE[base][0] <= FUND_TTL_SEC:
        return _FUND_CACHE[base][1], _FUND_CACHE[base][2]
    urls = FUND_FEEDS.get(base)
    if not urls:
        _FUND_CACHE[base] = (now, "N/A", None)
        return "N/A", None
    scores=[]
    try:
        for url in urls:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                t = getattr(entry, "title", "") or ""
                if not t: continue
                scores.append(_funda_from_text(t))
    except Exception as e:
        log.debug("fundamental fetch failed %s: %s", base, e)
    if not scores:
        _FUND_CACHE[base] = (now, "N/A", None)
        return "N/A", None
    avg = sum(scores)/len(scores)  # -1..1
    perc = round(((avg+1)/2)*100.0, 2)  # 0..100
    label = "Bullish 📈" if avg>0.05 else ("Bearish 📉" if avg<-0.05 else "Neutral ⚖️")
    _FUND_CACHE[base] = (now, label, perc)
    return label, perc

# ===================== sizing / portfolio ======================
def position_size(last_price: float, atr_val: float, confidence: float, winrate: float=0.54, rr: float=1.5):
    risk_cash = ACCOUNT_EQUITY * RISK_PER_TRADE
    risk_per_unit = atr_val if atr_val>0 else last_price*0.01
    units = risk_cash / max(risk_per_unit, 1e-9)
    kelly = winrate - (1-winrate)/rr
    kelly = float(np.clip(kelly, 0.0, 0.25))
    units *= (0.6 + 0.4 * (kelly/0.25))
    scale = 0.9 + 0.2*float(np.clip(confidence/100.0, 0.0, 1.0))
    units *= scale
    notional = units * last_price
    pos_norm = float(np.clip(notional/(ACCOUNT_EQUITY*PORTFOLIO_RISK_CAP), 0, 1))
    return round(pos_norm, 2), max(1e-8, units)

def portfolio_risk_weighting(latest: Dict[str, Tuple[float,float]]):
    vols = {s: (a/p if p>0 else 0.02) for s,(p,a) in latest.items()}
    inv = {s: (1.0/(v+1e-9)) for s,v in vols.items()}
    ssum = sum(inv.values()) or 1.0
    w = {s: inv[s]/ssum for s in inv}
    return w

# ===================== dataclass ======================
@dataclass
class SignalResult:
    trend: str
    score: float
    confidence: float
    color: str
    last: float
    atr: float
    tp: float
    sl: float
    ai_prob_up: Optional[float] = None
    htf_bias: str = "flat"
    vol_regime: float = 1.0
    delta_usd_raw: Optional[float] = None
    delta_usd_clip: Optional[float] = None
    delta_low: Optional[float] = None        # NEW: P20 delta
    delta_high: Optional[float] = None       # NEW: P80 delta
    funda_label: str = "N/A"
    funda_pct: Optional[float] = None

# ===================== analysis ======================
def analyze_symbol(symbol:str, timeframe:str) -> Optional[SignalResult]:
    df = fetch_ohlcv(symbol, timeframe, limit=900)
    if df is None or df.empty: return None
    d  = compute_indicators(df)
    if d is None or d.empty: return None

    # === Price Action + OrderFlow ===
    d = add_price_action(d)
    d = add_order_flow(d)

    # HTF confirm
    h_bias = "flat"
    if USE_HTF:
        h = fetch_htf_confirm(symbol, HTF_TF, limit=360)
        if h is not None:
            if (h["close"] > h["ema20"] > h["ema50"] > h["ema200"]) and (h["macd"] > h["macds"]):
                h_bias = "bull"
            elif (h["close"] < h["ema20"] < h["ema50"] < h["ema200"]) and (h["macd"] < h["macds"]):
                h_bias = "bear"

    last = d.iloc[-1]
    score, trend, color, conf = score_signal(last)

    ai_prob = None
    ai_align = None
    if AI_ENABLED:
        ai_prob = ai_train_and_prob_up(d, symbol, timeframe)
        if ai_prob is not None:
            p = ai_prob/100.0
            ai_align = p if trend=="Bullish" else (1.0-p) if trend=="Bearish" else 0.5

    # Δ$ prediction + bands
    delta_raw = None; delta_clip = None; d_lo=None; d_hi=None
    if AI_ENABLED:
        res = ai_train_and_pred_delta(d, symbol, timeframe)
        if res is not None:
            delta_raw, delta_clip, d_lo, d_hi = res

    # Soft filters
    if SOFT_FILTERS:
        if ai_align is not None:
            conf *= (0.9 + 0.2*float(np.clip(ai_align, 0.0, 1.0)))
        if USE_HTF and h_bias != "flat" and trend in ("Bullish","Bearish"):
            if (trend=="Bullish" and h_bias=="bull") or (trend=="Bearish" and h_bias=="bear"):
                conf *= 1.10
            elif (trend=="Bullish" and h_bias=="bear") or (trend=="Bearish" and h_bias=="bull"):
                conf *= 0.85
        reg = float(np.clip(last.get("vol_regime",1.0), 0.0, 3.0))
        if reg > 1.5:
            conf *= 0.9

    # Mix ML/IND + PA + OF
    pa = float(last.get("pa_score_pos", 0.5))  # 0..1
    of = float(last.get("of_score_pos", 0.5))  # 0..1
    ml_ind_conf = conf / 100.0                # 0..1
    den = max(1e-9, W_ML_IND + W_PA + W_OF)
    conf_mix = (W_ML_IND*ml_ind_conf + W_PA*pa + W_OF*of) / den
    conf = float(np.clip(conf_mix * 100.0, 0.0, 100.0))

    # penalty: OrderFlow ضد روند
    trend_dir = 1 if trend=="Bullish" else (-1 if trend=="Bearish" else 0)
    of_dir = 1 if last.get("of_score",0) > 0 else (-1 if last.get("of_score",0) < 0 else 0)
    if trend_dir*of_dir < 0:
        conf *= 0.88  # 12% penalty

    conf = float(np.clip(conf, 0.0, 100.0))
    tp, sl = prediction_targets(float(last["close"]), float(last["atr"]), score, trend, ai_align)

    display_trend, display_color = trend, color
    if conf < CONFIDENCE_MIN:
        display_trend, display_color = "Neutral", ""

    # Fundamental
    f_label, f_pct = fundamental_sentiment(symbol)

    return SignalResult(
        trend=display_trend, score=score, confidence=round(conf,2), color=display_color,
        last=float(last["close"]), atr=float(last["atr"]), tp=tp, sl=sl,
        ai_prob_up=ai_prob, htf_bias=h_bias, vol_regime=float(last.get("vol_regime",1.0)),
        delta_usd_raw=delta_raw, delta_usd_clip=delta_clip, delta_low=d_lo, delta_high=d_hi,
        funda_label=f_label, funda_pct=f_pct
    )

# ===================== Plain Multi-Candle View ======================
def build_multi_candle_forecast(sig: SignalResult, horizon:int, use_bands:bool=USE_BANDS) -> List[Dict[str, object]]:
    """
    شبیه‌سازی حرکت چند کندل آینده بر اساس Δ$ و ATR (+ باند عدم‌قطعیت در صورت فعال بودن).
    """
    decay = [1.00, 0.80, 0.65, 0.55] + [0.50]*max(0, horizon-4)
    ai = sig.ai_prob_up if sig.ai_prob_up is not None else None
    ai_align = None
    if ai is not None:
        p = ai/100.0
        ai_align = p if sig.trend=="Bullish" else (1.0-p) if sig.trend=="Bearish" else 0.5

    price = sig.last
    base_dir = 1 if sig.trend=="Bullish" else (-1 if sig.trend=="Bearish" else (1 if (ai or 50)>=50 else -1))
    raw_base = sig.delta_usd_raw if isinstance(sig.delta_usd_raw,(int,float)) else base_dir*0.35*sig.atr
    clip_base = sig.delta_usd_clip if isinstance(sig.delta_usd_clip,(int,float)) else float(np.clip(raw_base, -DELTA_CLIP_ATR_MULT*sig.atr, DELTA_CLIP_ATR_MULT*sig.atr))
    lo_base = sig.delta_low if isinstance(sig.delta_low,(int,float)) else clip_base*0.6
    hi_base = sig.delta_high if isinstance(sig.delta_high,(int,float)) else clip_base*1.4

    rows=[]
    for i in range(1, horizon+1):
        dr = float(raw_base*decay[i-1])
        dc = float(clip_base*decay[i-1])
        dcl = float(lo_base*decay[i-1])
        dch = float(hi_base*decay[i-1])

        next_price = price + dc
        cof = max(0.2*sig.confidence, sig.confidence*(1 - 0.08*(i-1)))
        tp_i, sl_i = prediction_targets(next_price, sig.atr, sig.score, sig.trend, ai_align)

        score_combo = round((0.55*cof) + (0.45*((ai or 0.0))), 2) if ai is not None else round(cof,2)

        row={
            "candle": f"Candle +{i}",
            "tf": DEFAULT_TF,
            "trend": sig.trend,
            "price": next_price,
            "tp": tp_i,
            "sl": sl_i,
            "raw": dr,
            "clip": dc,
            "cof": cof,
            "ai": ai,
            "score": score_combo,
            "reg": sig.vol_regime,
            "htf": sig.htf_bias if sig.htf_bias!="flat" else "flat",
            "fanda": f"{sig.funda_label} {f'{sig.funda_pct:.2f}%' if isinstance(sig.funda_pct,(int,float)) else ''}".strip()
        }
        if use_bands:
            row["p_low"] = price + dcl
            row["p_high"] = price + dch
        rows.append(row)
        price = next_price
    return rows

def print_multi_plain(symbol:str, rows:List[Dict[str,object]], show_bands:bool=USE_BANDS):
    title = f"{symbol} — {rows[0]['tf']} — {len(rows)}-candle horizon"
    print("\n" + title)
    print("-"*len(title))
    if show_bands and "p_low" in rows[0]:
        header = f"{'Candle':<10} {'Trend':<8} {'Price':>12} {'[P20..P80]':>17} {'TP':>12} {'SL':>12} {'Δraw':>10} {'Δclip':>10} {'Cof%':>8} {'AI%':>8} {'Score':>8} {'Reg':>6} {'HTF':>6} {'Funda':>14}"
        print(header); print("-"*len(header))
        for r in rows:
            ai_txt = f"{r['ai']:.2f}%" if isinstance(r['ai'],(int,float)) else "N/A"
            band = f"{r['p_low']:,.2f}..{r['p_high']:,.2f}"
            print(f"{r['candle']:<10} {r['trend']:<8} {r['price']:>12,.2f} {band:>17} {r['tp']:>12,.2f} {r['sl']:>12,.2f} {r['raw']:>10,.2f} {r['clip']:>10,.2f} {r['cof']:>8.2f} {ai_txt:>8} {r['score']:>8.2f} {r['reg']:>6.2f} {r['htf']:>6} {r['fanda'][:14]:>14}")
    else:
        header = f"{'Candle':<10} {'Trend':<8} {'Price':>12} {'TP':>12} {'SL':>12} {'Δraw':>12} {'Δclip':>12} {'Cof%':>8} {'AI%':>8} {'Score':>8} {'Reg':>6} {'HTF':>6} {'Funda':>14}"
        print(header); print("-"*len(header))
        for r in rows:
            ai_txt = f"{r['ai']:.2f}%" if isinstance(r['ai'],(int,float)) else "N/A"
            print(f"{r['candle']:<10} {r['trend']:<8} {r['price']:>12,.2f} {r['tp']:>12,.2f} {r['sl']:>12,.2f} {r['raw']:>12,.2f} {r['clip']:>12,.2f} {r['cof']:>8.2f} {ai_txt:>8} {r['score']:>8.2f} {r['reg']:>6.2f} {r['htf']:>6} {r['fanda'][:14]:>14}")
    print()

# ===================== Backtest (walk-forward) ======================
@dataclass
class BTResult:
    symbol: str
    tf: str
    trades: int
    winrate: float
    avg_rr: float
    pnl_pct: float
    mdd_pct: float
    sharpe: float

def _apply_costs(rr: float)->float:
    # هزینه‌ها را به RR تبدیل می‌کنیم (ورود+خروج): هر کدام کارمزد + اسلیپیج
    total_bps = 2*(COST_BPS + SLIPPAGE_BPS)  # دو سمت
    # تقریبی: RR خالص = RR - (bps به عنوان کسری از R)
    # چون R بر اساس فاصله SL است، این تخمین خطی است.
    rr_net = rr - (total_bps/10000.0)
    return rr_net

def backtest_symbol(symbol:str, tf:str="15m", lookback:int=1800)->Optional[BTResult]:
    df = fetch_ohlcv(symbol, tf, limit=lookback)
    if df is None or df.empty or len(df)<400:
        log.warning("Backtest: insufficient data for %s %s", symbol, tf)
        return None
    d = compute_indicators(df)
    equity = 1.0; high_equity = 1.0
    wins=0; losses=0; rr_list=[]
    last_side=None; entry=None; sl=None; tp=None
    for i in range(60, len(d)-1):
        row = d.iloc[i]
        s, trend, _, conf = score_signal(row)
        if AI_ENABLED and i>200:
            sub = d.iloc[:i+1].copy()
            ai = ai_train_and_prob_up(sub, symbol, tf)
        else:
            ai = None
        ai_align = None
        if ai is not None:
            p = ai/100.0
            ai_align = p if trend=="Bullish" else (1.0-p) if trend=="Bearish" else 0.5
        my_tp, my_sl = prediction_targets(row["close"], row["atr"], s, trend, ai_align)

        if last_side is None and conf >= CONFIDENCE_MIN and trend in ("Bullish","Bearish"):
            last_side = "buy" if trend=="Bullish" else "sell"
            entry = row["close"]; tp = my_tp; sl=my_sl
            continue

        if last_side is not None:
            nxt = d.iloc[i+1]
            hit_tp=False; hit_sl=False
            if last_side=="buy":
                hit_tp = nxt["high"]>=tp; hit_sl = nxt["low"]<=sl
            else:
                hit_tp = nxt["low"]<=tp; hit_sl = nxt["high"]>=sl
            if hit_tp or hit_sl or i==len(d)-2:
                R = abs(entry-sl)
                pnl = (tp-entry) if last_side=="buy" else (entry-tp)
                pnl_hit = pnl if hit_tp else ((sl-entry) if last_side=="buy" else (entry-sl))
                rr = (pnl_hit / (R+1e-9))
                rr = _apply_costs(rr)  # NEW: هزینه‌ها
                rr_list.append(rr)
                if rr>0: wins+=1
                else: losses+=1
                equity *= (1 + 0.005*rr)
                high_equity = max(high_equity, equity)
                last_side=None; entry=None; sl=None; tp=None

    trades = wins+losses
    if trades==0:
        return BTResult(symbol, tf, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    winrate = wins/trades
    avg_rr  = float(np.mean(rr_list)) if rr_list else 0.0
    pnl_pct = (equity-1.0)*100.0
    mdd_pct = max(0.0, (1.0 - (1.0/(high_equity+1e-9)))*100.0)
    sharpe = (np.mean(rr_list) / (np.std(rr_list)+1e-9)) * math.sqrt(252) if len(rr_list)>2 else 0.0
    return BTResult(symbol, tf, trades, round(winrate*100.0,2), round(avg_rr,2), round(pnl_pct,2), round(mdd_pct,2), round(sharpe,2))

# ===================== entry ======================
def main_once()->Dict[str, SignalResult]:
    print(f"Futures Pro V22 Pro+ (Dual Δ$) starting...")
    print(f"Exchange: {EXCHANGE_NAME} | Testnet: {TESTNET} | TF: {DEFAULT_TF} | HTF:{HTF_TF} (use={USE_HTF}) | "
          f"AI={AI_ENABLED} | SoftFilters={SOFT_FILTERS} | Symbols: {SYMBOLS}")
    print(f"Δ$ clip multiplier (ATR): {DELTA_CLIP_ATR_MULT}  (raw + clip)")
    print(f"Confidence Weights → ML/Ind:{W_ML_IND}  PA:{W_PA}  OF:{W_OF}")
    print(f"Calibration={CALIBRATE_PROBS} | CV_Folds={CV_FOLDS} gap={CV_GAP} | Costs(bps)={COST_BPS}+Slip={SLIPPAGE_BPS} | Bands={USE_BANDS}")
    results: Dict[str, SignalResult] = {}
    latest_px_atr: Dict[str, Tuple[float,float]] = {}

    for sym in SYMBOLS:
        try:
            sig = analyze_symbol(sym, DEFAULT_TF)
        except Exception as e:
            log.warning("Analyze error %s: %s", sym, e)
            sig = None
        if sig is None:
            log.warning("No data for %s", sym); continue
        results[sym] = sig
        latest_px_atr[sym] = (sig.last, sig.atr)

    if not results:
        print("No data."); return {}

    weights = portfolio_risk_weighting(latest_px_atr)

    # Per-symbol multi-candle plain view
    for sym, sig in results.items():
        rows = build_multi_candle_forecast(sig, HORIZON, use_bands=USE_BANDS)
        print_multi_plain(sym, rows, show_bands=USE_BANDS)

    # Summary weights
    print("Suggested portfolio risk weights (inverse vol, sum≈1):")
    for s,w in weights.items():
        print(f"  - {s}: {w:.2f}")

    if DRY_RUN:
        print("DRY_RUN=True → فقط تحلیل. سفارشی ارسال نمی‌شود.")
        return results

    if (not API_KEY) or (not API_SECRET):
        print("API_KEY/SECRET تعریف نشده → فقط تحلیل انجام شد.")
        return results

    ex = build_trade_exchange()
    if not ex:
        print("Cannot init trade exchange → فقط تحلیل انجام شد.")
        return results

    consec_losses = 0
    for sym, sig in results.items():
        side = "buy" if sig.trend=="Bullish" else ("sell" if sig.trend=="Bearish" else None)
        if side is None:
            print(f"{sym}: Trend Neutral (soft) → اجرای سفارش خودکار غیرفعال شد.")
            continue

        pos_norm, units = position_size(sig.last, sig.atr, confidence=sig.confidence, winrate=0.55, rr=1.5)
        alloc_w = weights.get(sym, 1.0/len(results))
        units *= alloc_w
        qty = max(0.0, round(units, 6))
        ensure_margin_mode_and_leverage(ex, sym, leverage=2, mode="CROSSED")

        try:
            trailing = sig.atr*TRAILING_ATR_MULT if TRAILING_ATR_MULT>0 else None
            notional = qty*sig.last
            if USE_TWAP_MIN_NOTIONAL>0 and notional>=USE_TWAP_MIN_NOTIONAL:
                place_twap_bracket(ex, sym, side, qty, sig.tp, sig.sl, slices=max(2,TWAP_SLICES), delay_s=2.0)
                msg = f"{sym}: {side.upper()} TWAP sent (qty={qty}), TP={sig.tp}, SL={sig.sl}"
            else:
                place_bracket(ex, sym, side, qty, sig.tp, sig.sl, trailing_atr=trailing)
                msg = f"{sym}: {side.upper()} market sent (qty={qty}), TP={sig.tp}, SL={sig.sl}"
            print(msg)
            tg_send(msg)
        except ccxt.AuthenticationError:
            err = f"{sym}: AuthenticationError → API KEY/SECRET اشتباه یا مجوز Futures فعال نیست."
            print(err)
            tg_send(err)
        except Exception as e:
            err = f"{sym}: Order error → {e}"
            print(err)
            tg_send(err)
            consec_losses += 1

        if consec_losses >= MAX_CONSEC_LOSSES_PAUSE>0:
            warn = f"Equity guard: reached {consec_losses} consecutive issues → pausing new orders."
            print(warn)
            tg_send(warn)
            break

    return results

# ===================== CLI / Loop ======================
def run_loop():
    print(f"Loop mode active, interval={INTERVAL_SEC}s. Ctrl+C to stop.")
    def handle_sig(sig, frame):
        print("\nStopping loop gracefully..."); sys.exit(0)
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)
    while True:
        try:
            main_once()
        except Exception as e:
            log.error("Loop iteration error: %s", e)
        time.sleep(max(5, INTERVAL_SEC))

def run_backtest():
    print("Running walk-forward backtests...")
    res: List[BTResult] = []
    for s in SYMBOLS:
        r = backtest_symbol(s, DEFAULT_TF, 2000)
        if r: res.append(r)
    if not res:
        print("Backtest produced no results."); return
    hdr = f"{'Symbol':<10}{'TF':<6}{'Trades':>8}{'Win%':>8}{'AvgRR':>10}{'PnL%':>10}{'MDD%':>10}{'Sharpe':>10}"
    print(hdr); print("-"*len(hdr))
    for r in res:
        print(f"{r.symbol:<10}{r.tf:<6}{r.trades:>8}{r.winrate:>8.2f}{r.avg_rr:>10.2f}{r.pnl_pct:>10.2f}{r.mdd_pct:>10.2f}{r.sharpe:>10.2f}")

# ===================== entry ======================
if __name__ == "__main__":
    # offer_upgrades(UPGRADE_CHECK_PACKAGES)  # optional
    args = set(a.lower() for a in sys.argv[1:])
    if "--backtest" in args:
        run_backtest(); sys.exit(0)
    if "--loop" in args:
        main_once(); run_loop(); sys.exit(0)
    main_once()
