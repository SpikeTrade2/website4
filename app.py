import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import pytz
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from enum import Enum
import base64

st.set_page_config(
    page_title="SpikeTrade - Penny Breakout Analysis",
    page_icon="📈",
    layout="wide"
)

TRADING_SETTINGS = {
    "buyPeriodMinutes": 12,
    "bbLengthMinutes": 10,
    "rsiLengthMinutes": 8,
    "priceRocPeriodMinutes": 20,
    "obvRocPeriodMinutes": 20,
    "mfiPeriodMinutes": 14,
    "mfiRocPeriodMinutes": 14,
    "vwapPeriodMinutes": 10,
    "dataPoints": 180,
    "spikePriceRocZThreshold": 1.0,
    "spikeRsiRocZThreshold": 0.5,
    "spikeObvRocZThreshold": 0.5,
    "spikeMfiRocZThreshold": 0.6,
    "spikePercentBRocZThreshold": 0.5,
    "spikeVwapRocZThreshold": 0.5,
    "spikeVolumeRocZThreshold": 0.5,
    "regularPriceRocThreshold": 2.0,
    "regularRsiRocThreshold": 5.0,
    "regularObvRocThreshold": 10.0,
    "regularMfiRocThreshold": 5.0,
    "regularPercentBRocThreshold": 15.0,
    "regularVwapRocThreshold": 1.5,
    "regularVolumeRocThreshold": 20.0,
    "comboSignalThreshold": 0.86,
    "highProbThreshold": 0.82,
    "stopLossPct": 0.02,
    "initialProfitFloorPct": 0.03,
    "subsequentProfitFloorPct": 0.02,
    "targetGainPercent": 2.0,
    "macdFastPeriod": 12,
    "macdSlowPeriod": 26,
    "macdSignalPeriod": 9,
    "macdHistogramRocThreshold": 0.5,
    "ema9Period": 9,
    "ema20Period": 20,
    "ema50Period": 50,
    "stochasticPeriod": 14,
    "stochasticKSmooth": 3,
    "stochasticDSmooth": 3,
    "stochasticOversoldThreshold": 30,
    "rvolPeriod": 20,
    "rvolThreshold": 1.2,
    "volumeSpikeThreshold": 1.5
}

CALIBRATED_WEIGHTS = {
    "price_roc": 0.44,
    "vwap": 0.25,
    "volume_spike": 0.49,
    "rsi_oversold": 0.85,
    "rvol_high": 0.44,
    "obv_roc": 0.51,
    "mfi": 0.67,
    "spike_quality": 0.48,
    "ema_downtrend": 0.46,
    "stoch_oversold": 0.47
}


class TradeSignal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NONE = "NONE"


@dataclass
class IndicatorsData:
    rsi: float = 50.0
    mfi: float = 50.0
    obv: float = 0.0
    vwap: float = 0.0
    bollingerUpper: float = 0.0
    bollingerLower: float = 0.0
    percentB: float = 0.5
    priceRoc: float = 0.0
    rsiRoc: float = 0.0
    obvRoc: float = 0.0
    mfiRoc: float = 0.0
    percentBRoc: float = 0.0
    vwapRoc: float = 0.0
    volumeRoc: float = 0.0
    macdLine: float = 0.0
    macdSignal: float = 0.0
    macdHistogram: float = 0.0
    macdHistogramRoc: float = 0.0
    ema9: float = 0.0
    ema20: float = 0.0
    ema50: float = 0.0
    stochK: float = 50.0
    stochD: float = 50.0
    rvol: float = 1.0
    volumeSpike: bool = False
    volumeRatio: float = 1.0
    atr: float = 0.0
    probability: float = 0.0
    confluenceScore: float = 0.0
    tradeSignal: TradeSignal = TradeSignal.NONE
    tradeSignalReason: str = ""


@dataclass
class AnalysisResult:
    symbol: str
    current_price: float
    indicators: IndicatorsData
    signal: TradeSignal
    probability: float
    reason: str
    predicted_profit_pct: float
    predicted_time_minutes: int
    prediction_confidence: float
    stop_loss_price: float
    target_price: float
    potential_profit_pct: float
    potential_loss_pct: float
    spikes_detected: List[str]


def get_logo_base64():
    logo_path = "static/logo2.jpg"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def apply_custom_css():
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        }
        
        .main-header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            padding: 20px;
            background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            border-radius: 15px;
            margin-bottom: 20px;
            border: 1px solid #0f3460;
            box-shadow: 0 4px 20px rgba(15, 52, 96, 0.3);
        }
        
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
        }
        
        .sub-header {
            text-align: center;
            color: #888;
            margin-bottom: 1.5rem;
            font-size: 1.1rem;
        }
        
        .card {
            background: linear-gradient(145deg, #1e1e2e 0%, #252538 100%);
            border: 1px solid #333;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        
        .card-title {
            font-size: 1.2rem;
            font-weight: bold;
            color: #00d4ff;
            margin-bottom: 15px;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }
        
        .buy-signal {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 12px;
            font-weight: bold;
            text-align: center;
            font-size: 1.5rem;
            box-shadow: 0 4px 20px rgba(40, 167, 69, 0.4);
            animation: pulse 2s infinite;
        }
        
        .sell-signal {
            background: linear-gradient(135deg, #dc3545 0%, #e74c3c 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 12px;
            font-weight: bold;
            text-align: center;
            font-size: 1.5rem;
            box-shadow: 0 4px 20px rgba(220, 53, 69, 0.4);
        }
        
        .hold-signal {
            background: linear-gradient(135deg, #ffc107 0%, #ffb300 100%);
            color: #1a1a1a;
            padding: 20px 30px;
            border-radius: 12px;
            font-weight: bold;
            text-align: center;
            font-size: 1.5rem;
            box-shadow: 0 4px 20px rgba(255, 193, 7, 0.4);
        }
        
        .none-signal {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 12px;
            font-weight: bold;
            text-align: center;
            font-size: 1.5rem;
            box-shadow: 0 4px 20px rgba(108, 117, 125, 0.4);
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 4px 20px rgba(40, 167, 69, 0.4); }
            50% { box-shadow: 0 4px 35px rgba(40, 167, 69, 0.7); }
            100% { box-shadow: 0 4px 20px rgba(40, 167, 69, 0.4); }
        }
        
        .probability-high { color: #00ff88; font-size: 3rem; font-weight: bold; text-shadow: 0 0 10px rgba(0, 255, 136, 0.5); }
        .probability-medium { color: #ffc107; font-size: 3rem; font-weight: bold; text-shadow: 0 0 10px rgba(255, 193, 7, 0.5); }
        .probability-low { color: #ff4757; font-size: 3rem; font-weight: bold; text-shadow: 0 0 10px rgba(255, 71, 87, 0.5); }
        
        .profit-positive { color: #00ff88; font-weight: bold; }
        .profit-negative { color: #ff4757; font-weight: bold; }
        
        .indicator-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            margin: 3px;
        }
        
        .badge-bullish {
            background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
            color: #1a1a1a;
        }
        
        .badge-bearish {
            background: linear-gradient(135deg, #ff4757 0%, #e74c3c 100%);
            color: white;
        }
        
        .badge-neutral {
            background: #333;
            color: #888;
        }
        
        .market-open {
            background: linear-gradient(135deg, #00ff88 0%, #28a745 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
        }
        
        .market-prepost {
            background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
            color: #1a1a1a;
            padding: 10px 20px;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
        }
        
        .market-closed {
            background: linear-gradient(135deg, #dc3545 0%, #c0392b 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
        }
        
        .time-estimate {
            background: linear-gradient(145deg, #1e1e2e 0%, #252538 100%);
            border: 1px solid #00d4ff;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        
        .time-value {
            font-size: 2rem;
            font-weight: bold;
            color: #00d4ff;
        }
        
        .stMetric {
            background: linear-gradient(145deg, #1e1e2e 0%, #252538 100%);
            border: 1px solid #333;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)


def get_market_status() -> Tuple[str, str]:
    et_tz = pytz.timezone('US/Eastern')
    now = datetime.now(et_tz)
    current_time = now.time()
    weekday = now.weekday()
    
    if weekday >= 5:
        return "closed", "Weekend - Market Closed"
    
    from datetime import time
    pre_market_start = time(4, 0)
    market_open = time(9, 30)
    market_close = time(16, 0)
    after_hours_end = time(20, 0)
    
    if current_time < pre_market_start:
        return "closed", "Market Closed"
    elif current_time < market_open:
        return "prepost", "Pre-Market Trading"
    elif current_time < market_close:
        return "open", "Market Open"
    elif current_time < after_hours_end:
        return "prepost", "After-Hours Trading"
    else:
        return "closed", "Market Closed"


class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)
        
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            else:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, 1)))
        return mfi
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        obv = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv
    
    @staticmethod
    def calculate_roc(series: pd.Series, period: int) -> pd.Series:
        return ((series - series.shift(period)) / series.shift(period).replace(0, 1)) * 100
    
    @staticmethod
    def calculate_z_score(series: pd.Series, lookback: int = 20) -> pd.Series:
        mean = series.rolling(window=lookback).mean()
        std = series.rolling(window=lookback).std()
        return (series - mean) / std.replace(0, 1)
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        percent_b = (prices - lower) / (upper - lower).replace(0, 1)
        return upper, middle, lower, percent_b
    
    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, 1)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum().replace(0, 1)
        return vwap
    
    @staticmethod
    def calculate_rvol(df: pd.DataFrame, period: int = 20) -> pd.Series:
        avg_volume = df['Volume'].rolling(window=period).mean()
        rvol = df['Volume'] / avg_volume.replace(0, 1)
        return rvol
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr


class ProbabilityCalibrator:
    def __init__(self):
        self.weights = self._load_calibrated_weights()
        self.intercept = -0.50
        self.probability_scale = 1.0
    
    def _load_calibrated_weights(self) -> Dict[str, float]:
        try:
            with open('indicator_calibration.json', 'r') as f:
                data = json.load(f)
                return {k: v.get('calibratedWeight', 0.5) for k, v in data.items()}
        except:
            return CALIBRATED_WEIGHTS
    
    def calculate_probability(self, indicators: IndicatorsData, spike_detected: bool) -> float:
        logit_score = self.intercept
        
        w = self.weights
        
        scaled_price_roc = max(-2.0, min(2.0, indicators.priceRoc / 2.0))
        logit_score += w.get('price_roc', 0.44) * scaled_price_roc
        
        if indicators.volumeSpike:
            logit_score += w.get('volume_spike', 0.49)
        
        if indicators.rsi < 30:
            logit_score += w.get('rsi_oversold', 0.85)
        elif indicators.rsi < 40:
            logit_score += w.get('rsi_oversold', 0.85) * 0.5
        
        if indicators.rvol > TRADING_SETTINGS['rvolThreshold']:
            rvol_score = min(1.0, (indicators.rvol - 1) / 1.0)
            logit_score += w.get('rvol_high', 0.44) * rvol_score
        
        if indicators.obvRoc > TRADING_SETTINGS['regularObvRocThreshold']:
            obv_score = min(1.0, indicators.obvRoc / (TRADING_SETTINGS['regularObvRocThreshold'] * 2))
            logit_score += w.get('obv_roc', 0.51) * obv_score
        elif indicators.obvRoc > 0:
            obv_score = indicators.obvRoc / TRADING_SETTINGS['regularObvRocThreshold']
            logit_score += w.get('obv_roc', 0.51) * obv_score * 0.5
        
        if indicators.mfi < 30:
            logit_score += w.get('mfi', 0.67)
        elif indicators.mfi < 40:
            logit_score += w.get('mfi', 0.67) * 0.6
        
        if indicators.stochK < TRADING_SETTINGS['stochasticOversoldThreshold']:
            logit_score += w.get('stoch_oversold', 0.47)
        
        current_price = indicators.ema9  # approximation
        if current_price > 0:
            price_below_emas = (current_price < indicators.ema9 and 
                               current_price < indicators.ema20 and 
                               current_price < indicators.ema50)
            if price_below_emas:
                logit_score -= w.get('ema_downtrend', 0.46)
        
        if spike_detected:
            logit_score += w.get('spike_quality', 0.48)
        
        scaled_vwap_roc = max(-2.0, min(2.0, indicators.vwapRoc / 2.0))
        logit_score += w.get('vwap', 0.25) * scaled_vwap_roc
        
        scaled_logit = logit_score * self.probability_scale
        probability = 1.0 / (1.0 + np.exp(-scaled_logit))
        return probability


class PredictionEngine:
    @staticmethod
    def estimate_time_to_target(indicators: IndicatorsData, target_gain_pct: float = 2.0) -> Tuple[int, float, str]:
        price_roc_per_min = indicators.priceRoc / 20 if indicators.priceRoc > 0 else 0
        volume_roc_per_min = indicators.volumeRoc / 20 if indicators.volumeRoc > 0 else 0
        
        avg_roc_per_min = (price_roc_per_min * 1.0 + volume_roc_per_min * 0.1) / 1.1
        
        if avg_roc_per_min <= 0:
            return 480, 0.3, "Low momentum - extended timeframe expected"
        
        minutes_per_percent = 1.0 / avg_roc_per_min
        estimated_minutes = int(minutes_per_percent * target_gain_pct)
        
        estimated_minutes = max(5, min(estimated_minutes, 1440))
        
        confidence = min(0.9, 0.3 + (indicators.probability * 0.4) + (0.2 if indicators.volumeSpike else 0))
        
        if estimated_minutes < 30:
            reason = "Strong momentum detected - rapid profit potential"
        elif estimated_minutes < 60:
            reason = "Moderate momentum - target within 1 hour"
        elif estimated_minutes < 240:
            reason = "Steady momentum - target within 4 hours"
        else:
            reason = "Extended timeframe - patience required"
        
        return estimated_minutes, confidence, reason
    
    @staticmethod
    def format_time(minutes: int) -> str:
        if minutes < 0:
            return "Unknown"
        elif minutes < 60:
            return f"{minutes} min"
        elif minutes < 1440:
            hours = minutes // 60
            mins = minutes % 60
            return f"{hours}h {mins}m"
        else:
            days = minutes // 1440
            hours = (minutes % 1440) // 60
            return f"{days}d {hours}h"


class PennyBreakoutAnalyzer:
    def __init__(self):
        self.settings = TRADING_SETTINGS
        self.calibrator = ProbabilityCalibrator()
        self.ti = TechnicalIndicators()
    
    def fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval="1m")
            if df.empty:
                df = ticker.history(period="5d", interval="5m")
            if df.empty:
                return None
            return df
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> IndicatorsData:
        indicators = IndicatorsData()
        
        if len(df) < 50:
            return indicators
        
        indicators.rsi = self.ti.calculate_rsi(df['Close'], self.settings['rsiLengthMinutes']).iloc[-1]
        indicators.mfi = self.ti.calculate_mfi(df, self.settings['mfiPeriodMinutes']).iloc[-1]
        
        obv_series = self.ti.calculate_obv(df)
        indicators.obv = obv_series.iloc[-1]
        
        indicators.priceRoc = self.ti.calculate_roc(df['Close'], self.settings['priceRocPeriodMinutes']).iloc[-1]
        indicators.volumeRoc = self.ti.calculate_roc(df['Volume'], self.settings['priceRocPeriodMinutes']).iloc[-1]
        
        rsi_series = self.ti.calculate_rsi(df['Close'], self.settings['rsiLengthMinutes'])
        indicators.rsiRoc = self.ti.calculate_roc(rsi_series, 14).iloc[-1]
        
        indicators.obvRoc = self.ti.calculate_roc(obv_series, self.settings['obvRocPeriodMinutes']).iloc[-1]
        
        mfi_series = self.ti.calculate_mfi(df, self.settings['mfiPeriodMinutes'])
        indicators.mfiRoc = self.ti.calculate_roc(mfi_series, 14).iloc[-1]
        
        bb_upper, bb_middle, bb_lower, percent_b = self.ti.calculate_bollinger_bands(
            df['Close'], self.settings['bbLengthMinutes']
        )
        indicators.bollingerUpper = bb_upper.iloc[-1]
        indicators.bollingerLower = bb_lower.iloc[-1]
        indicators.percentB = percent_b.iloc[-1]
        indicators.percentBRoc = self.ti.calculate_roc(percent_b, 20).iloc[-1]
        
        vwap_series = self.ti.calculate_vwap(df)
        indicators.vwap = vwap_series.iloc[-1]
        indicators.vwapRoc = self.ti.calculate_roc(vwap_series, self.settings['vwapPeriodMinutes']).iloc[-1]
        
        macd_line, macd_signal, macd_hist = self.ti.calculate_macd(
            df['Close'], 
            self.settings['macdFastPeriod'],
            self.settings['macdSlowPeriod'],
            self.settings['macdSignalPeriod']
        )
        indicators.macdLine = macd_line.iloc[-1]
        indicators.macdSignal = macd_signal.iloc[-1]
        indicators.macdHistogram = macd_hist.iloc[-1]
        
        if len(macd_hist) >= 2:
            delta_h = macd_hist.iloc[-1] - macd_hist.iloc[-2]
            current_price = df['Close'].iloc[-1]
            indicators.macdHistogramRoc = delta_h / current_price if current_price > 0 else 0
        
        indicators.ema9 = self.ti.calculate_ema(df['Close'], self.settings['ema9Period']).iloc[-1]
        indicators.ema20 = self.ti.calculate_ema(df['Close'], self.settings['ema20Period']).iloc[-1]
        indicators.ema50 = self.ti.calculate_ema(df['Close'], self.settings['ema50Period']).iloc[-1]
        
        stoch_k, stoch_d = self.ti.calculate_stochastic(
            df, 
            self.settings['stochasticPeriod'],
            self.settings['stochasticKSmooth']
        )
        indicators.stochK = stoch_k.iloc[-1]
        indicators.stochD = stoch_d.iloc[-1]
        
        indicators.rvol = self.ti.calculate_rvol(df, self.settings['rvolPeriod']).iloc[-1]
        
        avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        indicators.volumeRatio = current_volume / avg_volume if avg_volume > 0 else 1.0
        indicators.volumeSpike = indicators.volumeRatio > self.settings['volumeSpikeThreshold']
        
        indicators.atr = self.ti.calculate_atr(df).iloc[-1]
        
        return indicators
    
    def detect_spikes(self, df: pd.DataFrame, indicators: IndicatorsData) -> List[str]:
        spikes = []
        
        price_roc_z = self.ti.calculate_z_score(
            self.ti.calculate_roc(df['Close'], self.settings['priceRocPeriodMinutes'])
        ).iloc[-1]
        if price_roc_z > self.settings['spikePriceRocZThreshold']:
            spikes.append("Price ROC Spike")
        
        volume_roc_z = self.ti.calculate_z_score(
            self.ti.calculate_roc(df['Volume'], self.settings['priceRocPeriodMinutes'])
        ).iloc[-1]
        if volume_roc_z > self.settings['spikeVolumeRocZThreshold']:
            spikes.append("Volume ROC Spike")
        
        rsi_series = self.ti.calculate_rsi(df['Close'], self.settings['rsiLengthMinutes'])
        rsi_roc_z = self.ti.calculate_z_score(self.ti.calculate_roc(rsi_series, 14)).iloc[-1]
        if rsi_roc_z > self.settings['spikeRsiRocZThreshold']:
            spikes.append("RSI ROC Spike")
        
        obv_series = self.ti.calculate_obv(df)
        obv_roc_z = self.ti.calculate_z_score(
            self.ti.calculate_roc(obv_series, self.settings['obvRocPeriodMinutes'])
        ).iloc[-1]
        if obv_roc_z > self.settings['spikeObvRocZThreshold']:
            spikes.append("OBV ROC Spike")
        
        mfi_series = self.ti.calculate_mfi(df, self.settings['mfiPeriodMinutes'])
        mfi_roc_z = self.ti.calculate_z_score(self.ti.calculate_roc(mfi_series, 14)).iloc[-1]
        if mfi_roc_z > self.settings['spikeMfiRocZThreshold']:
            spikes.append("MFI ROC Spike")
        
        if indicators.volumeSpike:
            spikes.append("Volume Spike")
        
        return spikes
    
    def generate_signal(self, df: pd.DataFrame, indicators: IndicatorsData, spikes: List[str]) -> Tuple[TradeSignal, str]:
        current_price = df['Close'].iloc[-1]
        
        if len(spikes) == 0:
            return TradeSignal.HOLD, "No spike patterns detected - waiting for momentum"
        
        bullish_momentum = (
            indicators.priceRoc > 0 and 
            indicators.volumeRoc > 0 and 
            indicators.obvRoc > 0
        )
        if not bullish_momentum:
            return TradeSignal.HOLD, "Momentum not aligned - price, volume, or OBV declining"
        
        macd_threshold = self.settings['macdHistogramRocThreshold'] / 100.0
        macd_valid = indicators.macdHistogram <= 0 and indicators.macdHistogramRoc >= macd_threshold
        if not macd_valid:
            return TradeSignal.HOLD, "MACD histogram convergence not detected"
        
        stoch_oversold = indicators.stochK < self.settings['stochasticOversoldThreshold']
        if not stoch_oversold:
            return TradeSignal.HOLD, f"Stochastic not oversold (K={indicators.stochK:.1f})"
        
        rvol_valid = indicators.rvol > self.settings['rvolThreshold']
        if not rvol_valid:
            return TradeSignal.HOLD, f"RVOL too low ({indicators.rvol:.2f}x) - need unusual volume"
        
        price_below_all_emas = (
            current_price < indicators.ema9 and 
            current_price < indicators.ema20 and 
            current_price < indicators.ema50
        )
        if price_below_all_emas:
            return TradeSignal.SELL, "Price below all EMAs - strong downtrend detected"
        
        spike_detected = len(spikes) > 0
        probability = self.calibrator.calculate_probability(indicators, spike_detected)
        indicators.probability = probability
        
        if probability >= self.settings['comboSignalThreshold']:
            return TradeSignal.BUY, f"Strong buy signal - {len(spikes)} spike(s) detected with {probability*100:.1f}% probability"
        elif probability >= self.settings['highProbThreshold']:
            return TradeSignal.BUY, f"Buy signal - probability {probability*100:.1f}% meets threshold"
        else:
            return TradeSignal.HOLD, f"Signal probability ({probability*100:.1f}%) below threshold ({self.settings['highProbThreshold']*100:.0f}%)"
    
    def analyze(self, symbol: str) -> Optional[AnalysisResult]:
        symbol = symbol.upper().strip()
        
        df = self.fetch_data(symbol)
        if df is None or len(df) < 50:
            return None
        
        indicators = self.calculate_all_indicators(df)
        spikes = self.detect_spikes(df, indicators)
        signal, reason = self.generate_signal(df, indicators, spikes)
        
        spike_detected = len(spikes) > 0
        probability = self.calibrator.calculate_probability(indicators, spike_detected)
        indicators.probability = probability
        
        current_price = df['Close'].iloc[-1]
        target_gain = self.settings['targetGainPercent']
        stop_loss_pct = self.settings['stopLossPct'] * 100
        
        stop_loss_price = current_price * (1 - self.settings['stopLossPct'])
        target_price = current_price * (1 + target_gain / 100)
        
        predicted_minutes, confidence, pred_reason = PredictionEngine.estimate_time_to_target(
            indicators, target_gain
        )
        
        return AnalysisResult(
            symbol=symbol,
            current_price=current_price,
            indicators=indicators,
            signal=signal,
            probability=probability,
            reason=reason,
            predicted_profit_pct=target_gain,
            predicted_time_minutes=predicted_minutes,
            prediction_confidence=confidence,
            stop_loss_price=stop_loss_price,
            target_price=target_price,
            potential_profit_pct=target_gain,
            potential_loss_pct=stop_loss_pct,
            spikes_detected=spikes
        )


def create_probability_gauge(probability: float) -> go.Figure:
    if probability >= 0.8:
        color = "#00ff88"
    elif probability >= 0.6:
        color = "#ffc107"
    else:
        color = "#ff4757"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': '%', 'font': {'size': 40, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': color},
            'bgcolor': "#1a1a2e",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 60], 'color': 'rgba(255, 71, 87, 0.2)'},
                {'range': [60, 80], 'color': 'rgba(255, 193, 7, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(0, 255, 136, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#00d4ff", 'width': 4},
                'thickness': 0.75,
                'value': 82
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_price_chart(df: pd.DataFrame, result: AnalysisResult) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{result.symbol} - 1 Minute Chart', 'Volume')
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4757'
        ),
        row=1, col=1
    )
    
    ema9 = TechnicalIndicators.calculate_ema(df['Close'], 9)
    ema20 = TechnicalIndicators.calculate_ema(df['Close'], 20)
    
    fig.add_trace(
        go.Scatter(x=df.index, y=ema9, name='EMA 9', line=dict(color='#00d4ff', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=ema20, name='EMA 20', line=dict(color='#ffc107', width=1)),
        row=1, col=1
    )
    
    fig.add_hline(
        y=result.target_price, 
        line_dash="dash", 
        line_color="#00ff88",
        annotation_text=f"Target ${result.target_price:.2f}",
        row=1, col=1
    )
    fig.add_hline(
        y=result.stop_loss_price, 
        line_dash="dash", 
        line_color="#ff4757",
        annotation_text=f"Stop Loss ${result.stop_loss_price:.2f}",
        row=1, col=1
    )
    
    colors = ['#00ff88' if v > df['Volume'].rolling(20).mean().iloc[i] else '#4a4a5a' 
              for i, v in enumerate(df['Volume'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.8)',
        font={'color': '#ffffff'},
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(gridcolor='#333', showgrid=True)
    fig.update_yaxes(gridcolor='#333', showgrid=True)
    
    return fig


def main():
    apply_custom_css()
    
    logo_b64 = get_logo_base64()
    if logo_b64:
        st.markdown(f"""
        <div class="main-header-container">
            <img src="data:image/jpeg;base64,{logo_b64}" style="height: 60px; border-radius: 10px;">
            <h1 class="main-header">SpikeTrade</h1>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="main-header-container">
            <h1 class="main-header">📈 SpikeTrade</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<p class="sub-header">Penny Breakout Strategy Analysis</p>', unsafe_allow_html=True)
    
    market_status, market_text = get_market_status()
    status_class = f"market-{market_status}"
    st.markdown(f'<div class="{status_class}">{market_text}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        symbol = st.text_input(
            "Enter Stock Ticker Symbol",
            placeholder="e.g., AAPL, TSLA, GME",
            help="Enter a valid stock ticker symbol to analyze"
        )
    
    if symbol:
        with st.spinner(f"Analyzing {symbol.upper()}..."):
            analyzer = PennyBreakoutAnalyzer()
            result = analyzer.analyze(symbol)
        
        if result is None:
            st.error(f"Could not fetch data for {symbol.upper()}. Please check the ticker symbol.")
            return
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 Signal Probability</div>', unsafe_allow_html=True)
            
            gauge_fig = create_probability_gauge(result.probability)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🎯 Trade Recommendation</div>', unsafe_allow_html=True)
            
            if result.signal == TradeSignal.BUY:
                st.markdown(f'<div class="buy-signal">🟢 BUY</div>', unsafe_allow_html=True)
            elif result.signal == TradeSignal.SELL:
                st.markdown(f'<div class="sell-signal">🔴 SELL</div>', unsafe_allow_html=True)
            elif result.signal == TradeSignal.HOLD:
                st.markdown(f'<div class="hold-signal">🟡 HOLD</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="none-signal">⚪ NO SIGNAL</div>', unsafe_allow_html=True)
            
            st.markdown(f"<p style='text-align: center; color: #888; margin-top: 15px;'>{result.reason}</p>", 
                       unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Current Price",
                value=f"${result.current_price:.2f}"
            )
        
        with col2:
            st.metric(
                label="Target Price",
                value=f"${result.target_price:.2f}",
                delta=f"+{result.potential_profit_pct:.1f}%"
            )
        
        with col3:
            st.metric(
                label="Stop Loss",
                value=f"${result.stop_loss_price:.2f}",
                delta=f"-{result.potential_loss_pct:.1f}%"
            )
        
        with col4:
            time_str = PredictionEngine.format_time(result.predicted_time_minutes)
            st.markdown(f"""
            <div class="time-estimate">
                <div style="color: #888; font-size: 0.9rem;">Est. Time to Target</div>
                <div class="time-value">{time_str}</div>
                <div style="color: #888; font-size: 0.8rem;">Confidence: {result.prediction_confidence*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if result.spikes_detected:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">⚡ Detected Spike Patterns</div>', unsafe_allow_html=True)
            
            spike_html = ""
            for spike in result.spikes_detected:
                spike_html += f'<span class="indicator-badge badge-bullish">{spike}</span> '
            
            st.markdown(spike_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 Technical Indicators</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi_color = "badge-bullish" if result.indicators.rsi < 30 else ("badge-bearish" if result.indicators.rsi > 70 else "badge-neutral")
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #888;">RSI</div>
                <div class="indicator-badge {rsi_color}">{result.indicators.rsi:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            stoch_color = "badge-bullish" if result.indicators.stochK < 30 else ("badge-bearish" if result.indicators.stochK > 70 else "badge-neutral")
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #888;">Stochastic %K</div>
                <div class="indicator-badge {stoch_color}">{result.indicators.stochK:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            rvol_color = "badge-bullish" if result.indicators.rvol > 1.5 else ("badge-neutral" if result.indicators.rvol > 1.0 else "badge-bearish")
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #888;">Relative Volume</div>
                <div class="indicator-badge {rvol_color}">{result.indicators.rvol:.2f}x</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            macd_color = "badge-bullish" if result.indicators.macdHistogram > 0 else "badge-bearish"
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #888;">MACD Histogram</div>
                <div class="indicator-badge {macd_color}">{result.indicators.macdHistogram:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            roc_color = "badge-bullish" if result.indicators.priceRoc > 0 else "badge-bearish"
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #888;">Price ROC</div>
                <div class="indicator-badge {roc_color}">{result.indicators.priceRoc:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            vol_roc_color = "badge-bullish" if result.indicators.volumeRoc > 0 else "badge-bearish"
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #888;">Volume ROC</div>
                <div class="indicator-badge {vol_roc_color}">{result.indicators.volumeRoc:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            mfi_color = "badge-bullish" if result.indicators.mfi < 30 else ("badge-bearish" if result.indicators.mfi > 70 else "badge-neutral")
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #888;">MFI</div>
                <div class="indicator-badge {mfi_color}">{result.indicators.mfi:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            bb_color = "badge-bullish" if result.indicators.percentB < 0.3 else ("badge-bearish" if result.indicators.percentB > 0.7 else "badge-neutral")
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #888;">Bollinger %B</div>
                <div class="indicator-badge {bb_color}">{result.indicators.percentB:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📉 1-Minute Price Chart</div>', unsafe_allow_html=True)
        
        df = analyzer.fetch_data(symbol)
        if df is not None:
            chart_fig = create_price_chart(df.tail(100), result)
            st.plotly_chart(chart_fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander("📋 EMA Position Analysis"):
            current_price = result.current_price
            ema9 = result.indicators.ema9
            ema20 = result.indicators.ema20
            ema50 = result.indicators.ema50
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pos9 = "Above" if current_price > ema9 else "Below"
                color9 = "#00ff88" if current_price > ema9 else "#ff4757"
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 10px;">
                    <div style="color: #888;">EMA 9</div>
                    <div style="color: {color9}; font-size: 1.2rem; font-weight: bold;">${ema9:.2f}</div>
                    <div style="color: {color9};">{pos9}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                pos20 = "Above" if current_price > ema20 else "Below"
                color20 = "#00ff88" if current_price > ema20 else "#ff4757"
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 10px;">
                    <div style="color: #888;">EMA 20</div>
                    <div style="color: {color20}; font-size: 1.2rem; font-weight: bold;">${ema20:.2f}</div>
                    <div style="color: {color20};">{pos20}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                pos50 = "Above" if current_price > ema50 else "Below"
                color50 = "#00ff88" if current_price > ema50 else "#ff4757"
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 10px;">
                    <div style="color: #888;">EMA 50</div>
                    <div style="color: {color50}; font-size: 1.2rem; font-weight: bold;">${ema50:.2f}</div>
                    <div style="color: {color50};">{pos50}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #555; font-size: 0.8rem;">
        <p>SpikeTrade Penny Breakout Analysis | Data provided by Yahoo Finance</p>
        <p style="color: #ff4757;">⚠️ This is for educational purposes only. Not financial advice. Trade at your own risk.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
