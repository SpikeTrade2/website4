import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import pytz

st.set_page_config(
    page_title="Spiketrade",
    page_icon="📈",
    layout="wide"
)

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
        padding: 15px 25px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
        animation: pulse 2s infinite;
    }
    
    .sell-signal {
        background: linear-gradient(135deg, #dc3545 0%, #e74c3c 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.4);
    }
    
    .hold-signal {
        background: linear-gradient(135deg, #ffc107 0%, #ffb300 100%);
        color: #1a1a1a;
        padding: 15px 25px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 1.3rem;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.4);
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4); }
        50% { box-shadow: 0 4px 25px rgba(40, 167, 69, 0.7); }
        100% { box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4); }
    }
    
    .probability-high { color: #00ff88; font-size: 2.5rem; font-weight: bold; text-shadow: 0 0 10px rgba(0, 255, 136, 0.5); }
    .probability-medium { color: #ffc107; font-size: 2.5rem; font-weight: bold; text-shadow: 0 0 10px rgba(255, 193, 7, 0.5); }
    .probability-low { color: #ff4757; font-size: 2.5rem; font-weight: bold; text-shadow: 0 0 10px rgba(255, 71, 87, 0.5); }
    
    .trade-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #252538 100%);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .trade-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    }
    
    .trade-card-profit {
        border-left: 4px solid #00ff88;
    }
    
    .trade-card-loss {
        border-left: 4px solid #ff4757;
    }
    
    .trade-card-open {
        border-left: 4px solid #ffc107;
    }
    
    .trade-profit { color: #00ff88; font-weight: bold; font-size: 1.2rem; }
    .trade-loss { color: #ff4757; font-weight: bold; font-size: 1.2rem; }
    
    .indicator-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 3px;
    }
    
    .badge-on {
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        color: #1a1a1a;
    }
    
    .badge-off {
        background: #333;
        color: #666;
    }
    
    .filter-pass {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
    
    .filter-fail {
        background: linear-gradient(135deg, #dc3545 0%, #e74c3c 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
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
    
    .sidebar-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #252538 100%);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .sidebar-title {
        font-size: 1rem;
        font-weight: bold;
        color: #00d4ff;
        margin-bottom: 10px;
        border-bottom: 1px solid #333;
        padding-bottom: 8px;
    }
    
    .summary-stat {
        background: linear-gradient(145deg, #252538 0%, #1e1e2e 100%);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00d4ff;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #888;
        margin-top: 5px;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 8px 0;
        color: #ccc;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 4px;
    }
    
    .prob-breakdown {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    
    .prob-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 6px;
    }
    
    .prob-bar {
        height: 6px;
        background: linear-gradient(90deg, #00d4ff 0%, #00ff88 100%);
        border-radius: 3px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


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

TRADING_SETTINGS = {
    "buyPeriodMinutes": 48,
    "bbLengthMinutes": 24,
    "rsiLengthMinutes": 14,
    "priceRocPeriodMinutes": 20,
    "obvRocPeriodMinutes": 20,
    "mfiPeriodMinutes": 14,
    "vwapPeriodMinutes": 10,
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
    "comboSignalThreshold": 0.76,
    "highProbThreshold": 0.8,
    "stopLossPct": 0.02,
    "targetGainPercent": 2.0,
    "macdHistogramRocThreshold": 0.5,
    "stochasticOversoldThreshold": 30,
    "rvolThreshold": 1.2
}


def get_market_status():
    """Get current market status based on Eastern Time"""
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
        return "prepost", "Pre-Market"
    elif current_time < market_close:
        return "open", "Market Open"
    elif current_time < after_hours_end:
        return "prepost", "After-Hours"
    else:
        return "closed", "Market Closed"


class PennyBreakoutStrategy:
    def __init__(self):
        self.settings = TRADING_SETTINGS
        self.weights = CALIBRATED_WEIGHTS
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_mfi(self, df, period=14):
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
    
    def calculate_obv(self, df):
        obv = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv
    
    def calculate_roc(self, series, period):
        return ((series - series.shift(period)) / series.shift(period).replace(0, 1)) * 100
    
    def calculate_z_score(self, series, lookback=20):
        mean = series.rolling(window=lookback).mean()
        std = series.rolling(window=lookback).std()
        return (series - mean) / std.replace(0, 1)
    
    def calculate_ema(self, prices, period):
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        percent_b = (prices - lower) / (upper - lower)
        return upper, middle, lower, percent_b
    
    def calculate_stochastic(self, df, k_period=14, d_period=3):
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d
    
    def calculate_vwap(self, df):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap
    
    def calculate_rvol(self, df, period=20):
        avg_volume = df['Volume'].rolling(window=period).mean()
        rvol = df['Volume'] / avg_volume.replace(0, 1)
        return rvol
    
    def calculate_signals(self, df):
        df = df.copy()
        
        df['RSI'] = self.calculate_rsi(df['Close'], self.settings['rsiLengthMinutes'])
        df['MFI'] = self.calculate_mfi(df, self.settings['mfiPeriodMinutes'])
        df['OBV'] = self.calculate_obv(df)
        
        df['Price_ROC'] = self.calculate_roc(df['Close'], self.settings['priceRocPeriodMinutes'])
        df['Volume_ROC'] = self.calculate_roc(df['Volume'], self.settings['priceRocPeriodMinutes'])
        df['RSI_ROC'] = self.calculate_roc(df['RSI'], self.settings['rsiLengthMinutes'])
        df['OBV_ROC'] = self.calculate_roc(df['OBV'], self.settings['obvRocPeriodMinutes'])
        df['MFI_ROC'] = self.calculate_roc(df['MFI'], self.settings['mfiPeriodMinutes'])
        
        df['Price_ROC_Z'] = self.calculate_z_score(df['Price_ROC'])
        df['Volume_ROC_Z'] = self.calculate_z_score(df['Volume_ROC'])
        df['RSI_ROC_Z'] = self.calculate_z_score(df['RSI_ROC'])
        df['OBV_ROC_Z'] = self.calculate_z_score(df['OBV_ROC'])
        df['MFI_ROC_Z'] = self.calculate_z_score(df['MFI_ROC'])
        
        df['EMA_9'] = self.calculate_ema(df['Close'], 9)
        df['EMA_20'] = self.calculate_ema(df['Close'], 20)
        df['EMA_50'] = self.calculate_ema(df['Close'], 50)
        
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(df['Close'])
        df['MACD_Hist_ROC'] = df['MACD_Hist'].diff() / df['Close']
        
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'], df['Percent_B'] = self.calculate_bollinger_bands(df['Close'], self.settings['bbLengthMinutes'])
        df['Percent_B_ROC'] = self.calculate_roc(df['Percent_B'], self.settings['priceRocPeriodMinutes'])
        df['Percent_B_ROC_Z'] = self.calculate_z_score(df['Percent_B_ROC'])
        
        df['Stoch_K'], df['Stoch_D'] = self.calculate_stochastic(df)
        
        df['VWAP'] = self.calculate_vwap(df)
        df['VWAP_ROC'] = self.calculate_roc(df['VWAP'], self.settings['vwapPeriodMinutes'])
        df['VWAP_ROC_Z'] = self.calculate_z_score(df['VWAP_ROC'])
        
        df['RVOL'] = self.calculate_rvol(df)
        
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Spike'] = df['Volume'] > (df['Volume_MA'] * 1.5)
        
        df['Price_Spike'] = df['Price_ROC_Z'] > self.settings['spikePriceRocZThreshold']
        df['RSI_Spike'] = df['RSI_ROC_Z'] > self.settings['spikeRsiRocZThreshold']
        df['OBV_Spike'] = df['OBV_ROC_Z'] > self.settings['spikeObvRocZThreshold']
        df['MFI_Spike'] = df['MFI_ROC_Z'] > self.settings['spikeMfiRocZThreshold']
        df['Vol_ROC_Spike'] = df['Volume_ROC_Z'] > self.settings['spikeVolumeRocZThreshold']
        df['Percent_B_Spike'] = df['Percent_B_ROC_Z'] > self.settings['spikePercentBRocZThreshold']
        df['VWAP_Spike'] = df['VWAP_ROC_Z'] > self.settings['spikeVwapRocZThreshold']
        
        df = self.generate_signals_with_one_trade(df)
        
        return df
    
    def generate_signals_with_one_trade(self, df):
        df['Buy_Signal'] = False
        df['Sell_Signal'] = False
        df['Signal_Probability'] = 0.0
        
        in_trade = False
        entry_price = 0.0
        entry_idx = None
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if not in_trade:
                probability, reasons = self.calculate_buy_probability(df, i)
                df.iloc[i, df.columns.get_loc('Signal_Probability')] = probability
                
                has_spike = (row['Price_Spike'] or row['RSI_Spike'] or 
                           row['OBV_Spike'] or row['MFI_Spike'] or row['Vol_ROC_Spike'] or
                           row['Percent_B_Spike'] or row['VWAP_Spike'])
                
                if not has_spike:
                    continue
                
                bullish_momentum = row['Price_ROC'] > 0 and row['Volume_ROC'] > 0 and row['OBV_ROC'] > 0
                if not bullish_momentum:
                    continue
                
                macd_hist_roc_threshold = self.settings['macdHistogramRocThreshold'] / 100.0
                macd_histogram_roc_valid = row['MACD_Hist'] <= 0 and row['MACD_Hist_ROC'] >= macd_hist_roc_threshold
                if not macd_histogram_roc_valid:
                    continue
                
                stoch_oversold = row['Stoch_K'] < self.settings['stochasticOversoldThreshold']
                if not stoch_oversold:
                    continue
                
                rvol_valid = row['RVOL'] > self.settings['rvolThreshold']
                if not rvol_valid:
                    continue
                
                price_below_all_emas = (row['Close'] < row['EMA_9'] and 
                                        row['Close'] < row['EMA_20'] and 
                                        row['Close'] < row['EMA_50'])
                if price_below_all_emas:
                    continue
                
                if probability >= self.settings['comboSignalThreshold']:
                    df.iloc[i, df.columns.get_loc('Buy_Signal')] = True
                    in_trade = True
                    entry_price = row['Close']
                    entry_idx = i
            
            else:
                current_price = row['Close']
                pnl_pct = (current_price - entry_price) / entry_price
                
                hit_stop_loss = pnl_pct <= -self.settings['stopLossPct']
                hit_target = pnl_pct >= self.settings['targetGainPercent'] / 100
                
                macd_bearish = row['MACD'] < row['MACD_Signal'] and prev_row['MACD'] >= prev_row['MACD_Signal']
                
                rsi_overbought = row['RSI'] > 70
                
                below_all_emas = (row['Close'] < row['EMA_9'] and 
                                 row['Close'] < row['EMA_20'] and 
                                 row['Close'] < row['EMA_50'])
                
                bearish_momentum = row['Price_ROC'] < -1 and row['Volume_Spike']
                
                if hit_stop_loss or hit_target or macd_bearish or (rsi_overbought and bearish_momentum) or below_all_emas:
                    df.iloc[i, df.columns.get_loc('Sell_Signal')] = True
                    in_trade = False
                    entry_price = 0.0
                    entry_idx = None
        
        return df
    
    def calculate_buy_probability(self, df, idx):
        if idx < 50:
            return 0.0, {}
        
        row = df.iloc[idx]
        scores = {}
        total_weight = 0.0
        weighted_score = 0.0
        
        if row['Price_ROC'] > 0:
            score = min(1.0, row['Price_ROC'] / self.settings['regularPriceRocThreshold'])
            weight = self.weights['price_roc']
            weighted_score += score * weight
            total_weight += weight
            scores['Price ROC'] = f"+{score*100:.0f}%"
        
        if row['Volume_Spike']:
            weight = self.weights['volume_spike']
            weighted_score += weight
            total_weight += weight
            scores['Volume Spike'] = "Active"
        
        if row['RSI'] < 30:
            weight = self.weights['rsi_oversold']
            weighted_score += weight
            total_weight += weight
            scores['RSI Oversold'] = f"{row['RSI']:.1f}"
        elif row['RSI'] < 40:
            weight = self.weights['rsi_oversold'] * 0.5
            weighted_score += weight
            total_weight += weight
            scores['RSI Low'] = f"{row['RSI']:.1f}"
        
        if row['RVOL'] > self.settings['rvolThreshold']:
            score = min(1.0, (row['RVOL'] - 1) / 1.0)
            weight = self.weights['rvol_high']
            weighted_score += score * weight
            total_weight += weight
            scores['RVOL High'] = f"{row['RVOL']:.2f}x"
        
        if row['OBV_ROC'] > self.settings['regularObvRocThreshold']:
            score = min(1.0, row['OBV_ROC'] / (self.settings['regularObvRocThreshold'] * 2))
            weight = self.weights['obv_roc']
            weighted_score += score * weight
            total_weight += weight
            scores['OBV ROC'] = f"+{row['OBV_ROC']:.1f}%"
        elif row['OBV_ROC'] > 0:
            score = row['OBV_ROC'] / self.settings['regularObvRocThreshold']
            weight = self.weights['obv_roc'] * 0.5
            weighted_score += score * weight
            total_weight += weight
            scores['OBV ROC'] = f"+{row['OBV_ROC']:.1f}%"
        
        if row['MFI'] < 30:
            weight = self.weights['mfi']
            weighted_score += weight
            total_weight += weight
            scores['MFI Oversold'] = f"{row['MFI']:.1f}"
        elif row['MFI'] < 40:
            weight = self.weights['mfi'] * 0.6
            weighted_score += weight
            total_weight += weight
            scores['MFI Low'] = f"{row['MFI']:.1f}"
        
        if row['MFI_ROC'] > self.settings['regularMfiRocThreshold']:
            score = min(1.0, row['MFI_ROC'] / (self.settings['regularMfiRocThreshold'] * 2))
            weight = self.weights['mfi'] * 0.5
            weighted_score += score * weight
            total_weight += weight
            scores['MFI ROC'] = f"+{row['MFI_ROC']:.1f}%"
        
        if row['Stoch_K'] < self.settings['stochasticOversoldThreshold']:
            score = 1.0 - (row['Stoch_K'] / self.settings['stochasticOversoldThreshold'])
            weight = self.weights['stoch_oversold']
            weighted_score += score * weight
            total_weight += weight
            scores['Stoch Oversold'] = f"{row['Stoch_K']:.1f}"
        
        if row['Close'] > row['EMA_9'] and row['Close'] > row['EMA_20']:
            weight = self.weights['ema_downtrend']
            weighted_score += weight
            total_weight += weight
            scores['Above EMAs'] = "Yes"
        elif row['Close'] > row['EMA_9']:
            weight = self.weights['ema_downtrend'] * 0.5
            weighted_score += weight
            total_weight += weight
            scores['Above EMA9'] = "Yes"
        
        spike_count = sum([row['Price_Spike'], row['RSI_Spike'], row['OBV_Spike'], 
                          row['MFI_Spike'], row['Vol_ROC_Spike'], row['Percent_B_Spike'],
                          row['VWAP_Spike']])
        if spike_count >= 2:
            score = min(1.0, spike_count / 4.0)
            weight = self.weights['spike_quality']
            weighted_score += score * weight
            total_weight += weight
            scores['Multi-Spike'] = f"{spike_count} spikes"
        
        if row['Close'] > row['VWAP']:
            vwap_distance = (row['Close'] - row['VWAP']) / row['VWAP'] * 100
            score = min(1.0, vwap_distance / 1.0)
            weight = self.weights['vwap']
            weighted_score += score * weight
            total_weight += weight
            scores['Above VWAP'] = f"+{vwap_distance:.2f}%"
        
        if row['Percent_B_ROC'] > self.settings['regularPercentBRocThreshold']:
            score = min(1.0, row['Percent_B_ROC'] / (self.settings['regularPercentBRocThreshold'] * 2))
            weight = self.weights['spike_quality'] * 0.3
            weighted_score += score * weight
            total_weight += weight
            scores['%B ROC'] = f"+{row['Percent_B_ROC']:.1f}%"
        
        if row['VWAP_ROC'] > self.settings['regularVwapRocThreshold']:
            score = min(1.0, row['VWAP_ROC'] / (self.settings['regularVwapRocThreshold'] * 2))
            weight = self.weights['vwap'] * 0.5
            weighted_score += score * weight
            total_weight += weight
            scores['VWAP ROC'] = f"+{row['VWAP_ROC']:.2f}%"
        
        if total_weight > 0:
            probability = weighted_score / total_weight
        else:
            probability = 0.0
        
        probability = max(0.0, min(1.0, probability))
        
        return probability, scores
    
    def get_current_signal(self, df):
        if len(df) < 2:
            return "HOLD", "Insufficient data", 0.0
        
        latest = df.iloc[-1]
        
        if latest['Buy_Signal']:
            return "BUY", "Buy signal triggered", latest['Signal_Probability']
        elif latest['Sell_Signal']:
            return "SELL", "Sell signal triggered", 0.0
        
        probability, _ = self.calculate_buy_probability(df, len(df)-1)
        return "HOLD", "Monitoring for signals", probability
    
    def get_filter_status(self, df):
        """Get the status of each filter for the latest candle"""
        if len(df) < 51:
            return {}
        
        row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else row
        
        has_spike = (row['Price_Spike'] or row['RSI_Spike'] or 
                   row['OBV_Spike'] or row['MFI_Spike'] or row['Vol_ROC_Spike'] or
                   row['Percent_B_Spike'] or row['VWAP_Spike'])
        
        bullish_momentum = row['Price_ROC'] > 0 and row['Volume_ROC'] > 0 and row['OBV_ROC'] > 0
        
        macd_hist_roc_threshold = self.settings['macdHistogramRocThreshold'] / 100.0
        macd_valid = row['MACD_Hist'] <= 0 and row['MACD_Hist_ROC'] >= macd_hist_roc_threshold
        
        stoch_oversold = row['Stoch_K'] < self.settings['stochasticOversoldThreshold']
        
        rvol_valid = row['RVOL'] > self.settings['rvolThreshold']
        
        price_above_ema = not (row['Close'] < row['EMA_9'] and 
                              row['Close'] < row['EMA_20'] and 
                              row['Close'] < row['EMA_50'])
        
        return {
            'has_spike': has_spike,
            'bullish_momentum': bullish_momentum,
            'macd_gate': macd_valid,
            'stochastic': stoch_oversold,
            'rvol': rvol_valid,
            'ema_trend': price_above_ema
        }
    
    def get_spike_status(self, df):
        """Get the status of each spike detector"""
        if len(df) < 51:
            return {}
        
        row = df.iloc[-1]
        return {
            'Price': bool(row['Price_Spike']),
            'RSI': bool(row['RSI_Spike']),
            'OBV': bool(row['OBV_Spike']),
            'MFI': bool(row['MFI_Spike']),
            'Volume': bool(row['Vol_ROC_Spike']),
            '%B': bool(row['Percent_B_Spike']),
            'VWAP': bool(row['VWAP_Spike'])
        }
    
    def get_paired_trades(self, df):
        trades = []
        buy_signals = df[df['Buy_Signal']]
        sell_signals = df[df['Sell_Signal']]
        
        for buy_idx, buy_row in buy_signals.iterrows():
            sell_after = sell_signals[sell_signals.index > buy_idx]
            if not sell_after.empty:
                sell_idx = sell_after.index[0]
                sell_row = df.loc[sell_idx]
                
                entry_price = buy_row['Close']
                exit_price = sell_row['Close']
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                
                trades.append({
                    'entry_time': buy_idx,
                    'exit_time': sell_idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'probability': buy_row['Signal_Probability']
                })
            else:
                trades.append({
                    'entry_time': buy_idx,
                    'exit_time': None,
                    'entry_price': buy_row['Close'],
                    'exit_price': None,
                    'pnl_pct': None,
                    'probability': buy_row['Signal_Probability'],
                    'open': True
                })
        
        return trades


def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1d", interval="1m", prepost=True)
        if df.empty:
            df = stock.history(period="2d", interval="1m", prepost=True)
            if not df.empty:
                today = datetime.now().date()
                df = df[df.index.date == today]
        if df.empty:
            return None, "No data available for this ticker today"
        return df, None
    except Exception as e:
        return None, str(e)


def create_chart(df, ticker, trades):
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4, 0.1, 0.15, 0.15, 0.2],
        subplot_titles=(f'{ticker} - 1 Day / 1 Minute (Pre/Post Market)', 'Signal Probability', 'Volume & RVOL', 'RSI & Stochastic', 'MACD')
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_9'], name='EMA 9', line=dict(color='#FF9800', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', line=dict(color='#9C27B0', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_50'], name='EMA 50', line=dict(color='#2196F3', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='#00BCD4', width=2, dash='dot')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash'),
                   fill='tonexty', fillcolor='rgba(128,128,128,0.08)'),
        row=1, col=1
    )
    
    if len(df) > 0:
        latest = df.iloc[-1]
        last_time = df.index[-1]
        
        fig.add_annotation(
            x=last_time, y=latest['VWAP'],
            text=f"VWAP: ${latest['VWAP']:.2f}",
            showarrow=False, xanchor='left', font=dict(color='#00BCD4', size=10),
            row=1, col=1
        )
        fig.add_annotation(
            x=last_time, y=latest['EMA_9'],
            text=f"EMA9: ${latest['EMA_9']:.2f}",
            showarrow=False, xanchor='left', font=dict(color='#FF9800', size=10),
            row=1, col=1
        )
        fig.add_annotation(
            x=last_time, y=latest['EMA_20'],
            text=f"EMA20: ${latest['EMA_20']:.2f}",
            showarrow=False, xanchor='left', font=dict(color='#9C27B0', size=10),
            row=1, col=1
        )
    
    for trade in trades:
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        
        fig.add_trace(
            go.Scatter(
                x=[entry_time],
                y=[entry_price * 0.997],
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=18, color='#00ff88', line=dict(color='#004d26', width=2)),
                text=['BUY'],
                textposition='bottom center',
                textfont=dict(color='#00ff88', size=10, family='Arial Black'),
                name='Buy',
                showlegend=False
            ),
            row=1, col=1
        )
        
        if trade.get('exit_time'):
            exit_time = trade['exit_time']
            exit_price = trade['exit_price']
            
            fig.add_trace(
                go.Scatter(
                    x=[exit_time],
                    y=[exit_price * 1.003],
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=18, color='#ff4757', line=dict(color='#8b0000', width=2)),
                    text=['SELL'],
                    textposition='top center',
                    textfont=dict(color='#ff4757', size=10, family='Arial Black'),
                    name='Sell',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            color = 'rgba(0,255,136,0.15)' if trade['pnl_pct'] > 0 else 'rgba(255,71,87,0.15)'
            border_color = 'rgba(0,255,136,0.5)' if trade['pnl_pct'] > 0 else 'rgba(255,71,87,0.5)'
            fig.add_shape(
                type="rect",
                x0=entry_time, x1=exit_time,
                y0=min(entry_price, exit_price) * 0.995,
                y1=max(entry_price, exit_price) * 1.005,
                fillcolor=color,
                line=dict(color=border_color, width=1),
                row=1, col=1
            )
    
    prob_colors = ['#00ff88' if p >= 0.76 else '#ffc107' if p >= 0.5 else '#ff4757' for p in df['Signal_Probability']]
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['Signal_Probability'] * 100,
            name='Buy Probability',
            line=dict(color='#00d4ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.2)'
        ),
        row=2, col=1
    )
    fig.add_hline(y=76, line_dash="dash", line_color="#00ff88", line_width=1, row=2, col=1)
    fig.add_annotation(x=df.index[-1], y=76, text="Signal Threshold (76%)", showarrow=False, 
                      xanchor='left', font=dict(color='#00ff88', size=9), row=2, col=1)
    
    colors = ['#26a69a' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ef5350' for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.7),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Volume_MA'], name='Vol MA', line=dict(color='white', width=1)),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#2196F3', width=2)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Stoch_K'], name='Stoch K', line=dict(color='#FF9800', width=1.5)),
        row=4, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#ff4757", line_width=1, row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00ff88", line_width=1, row=4, col=1)
    
    colors_macd = ['#26a69a' if val >= 0 else '#ef5350' for val in df['MACD_Hist']]
    fig.add_trace(
        go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist', marker_color=colors_macd),
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#2196F3', width=1.5)),
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='#FF9800', width=1.5)),
        row=5, col=1
    )
    
    fig.update_layout(
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color='white', size=10)
        ),
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,10,10,0.8)',
        font=dict(color='white'),
        margin=dict(l=60, r=60, t=80, b=40)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(50,50,50,0.5)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(50,50,50,0.5)')
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Prob %", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    fig.update_yaxes(title_text="RSI/Stoch", row=4, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=5, col=1)
    
    return fig


def render_sidebar(strategy, df=None):
    """Render the sidebar with market status, quick tickers, settings, and legend"""
    with st.sidebar:
        st.image("logo2.jpg", width=120)
        st.markdown("---")
        
        market_status, market_text = get_market_status()
        st.markdown('<div class="sidebar-title">📊 Market Status</div>', unsafe_allow_html=True)
        if market_status == "open":
            st.markdown(f'<div class="market-open">🟢 {market_text}</div>', unsafe_allow_html=True)
        elif market_status == "prepost":
            st.markdown(f'<div class="market-prepost">🟡 {market_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="market-closed">🔴 {market_text}</div>', unsafe_allow_html=True)
        
        et_tz = pytz.timezone('US/Eastern')
        current_et = datetime.now(et_tz).strftime('%H:%M ET')
        st.caption(f"Current Time: {current_et}")
        
        st.markdown("---")
        st.markdown('<div class="sidebar-title">⚡ Quick Tickers</div>', unsafe_allow_html=True)
        
        penny_stocks = ["MULN", "SNDL", "BBIG", "CLOV", "SOFI", "PLTR"]
        cols = st.columns(3)
        for i, ticker in enumerate(penny_stocks):
            with cols[i % 3]:
                if st.button(ticker, key=f"quick_{ticker}", use_container_width=True):
                    st.session_state['selected_ticker'] = ticker
        
        st.markdown("---")
        st.markdown('<div class="sidebar-title">⚙️ Strategy Settings</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="sidebar-card">
            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                <span style="color: #888;">Signal Threshold:</span>
                <span style="color: #00d4ff; font-weight: bold;">{strategy.settings['comboSignalThreshold']*100:.0f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                <span style="color: #888;">Stop Loss:</span>
                <span style="color: #ff4757; font-weight: bold;">{strategy.settings['stopLossPct']*100:.0f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                <span style="color: #888;">Profit Target:</span>
                <span style="color: #00ff88; font-weight: bold;">{strategy.settings['targetGainPercent']:.0f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                <span style="color: #888;">RVOL Threshold:</span>
                <span style="color: #ffc107; font-weight: bold;">{strategy.settings['rvolThreshold']}x</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                <span style="color: #888;">Stoch Oversold:</span>
                <span style="color: #ffc107; font-weight: bold;">&lt;{strategy.settings['stochasticOversoldThreshold']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-title">📖 Signal Legend</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(135deg, #00ff88, #28a745);"></div>
                <span><b>BUY</b> - All conditions met</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(135deg, #ff4757, #dc3545);"></div>
                <span><b>SELL</b> - Exit triggered</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(135deg, #ffc107, #ff9800);"></div>
                <span><b>HOLD</b> - Monitoring</span>
            </div>
            <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #333;">
                <div class="legend-item">
                    <span style="color: #00d4ff;">▲</span>
                    <span>Entry Point</span>
                </div>
                <div class="legend-item">
                    <span style="color: #ff4757;">▼</span>
                    <span>Exit Point</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_indicator_dashboard(strategy, df):
    """Render the comprehensive indicator dashboard"""
    if df is None or len(df) < 51:
        return
    
    st.markdown("---")
    
    spike_status = strategy.get_spike_status(df)
    filter_status = strategy.get_filter_status(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card"><div class="card-title">🔥 Spike Detectors (7)</div>', unsafe_allow_html=True)
        
        spike_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
        for name, active in spike_status.items():
            badge_class = "badge-on" if active else "badge-off"
            icon = "✓" if active else "○"
            spike_html += f'<span class="indicator-badge {badge_class}">{icon} {name}</span>'
        spike_html += '</div>'
        
        active_count = sum(spike_status.values())
        spike_html += f'<div style="margin-top: 15px; color: #888;">Active Spikes: <span style="color: #00d4ff; font-weight: bold;">{active_count}/7</span></div>'
        spike_html += '</div>'
        st.markdown(spike_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card"><div class="card-title">🎯 Signal Filters</div>', unsafe_allow_html=True)
        
        filters = [
            ('MACD Gate', filter_status.get('macd_gate', False)),
            ('Stochastic', filter_status.get('stochastic', False)),
            ('RVOL', filter_status.get('rvol', False)),
            ('EMA Trend', filter_status.get('ema_trend', False)),
            ('Has Spike', filter_status.get('has_spike', False)),
            ('Momentum', filter_status.get('bullish_momentum', False))
        ]
        
        filter_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
        for name, passed in filters:
            if passed:
                filter_html += f'<span class="filter-pass">✓ {name}</span>'
            else:
                filter_html += f'<span class="filter-fail">✗ {name}</span>'
        filter_html += '</div>'
        
        pass_count = sum(1 for _, passed in filters if passed)
        filter_html += f'<div style="margin-top: 15px; color: #888;">Filters Passing: <span style="color: #00d4ff; font-weight: bold;">{pass_count}/{len(filters)}</span></div>'
        filter_html += '</div>'
        st.markdown(filter_html, unsafe_allow_html=True)


def render_probability_breakdown(strategy, df):
    """Render the weighted probability breakdown"""
    if df is None or len(df) < 51:
        return
    
    probability, scores = strategy.calculate_buy_probability(df, len(df)-1)
    
    if scores:
        with st.expander("📊 Probability Breakdown", expanded=False):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            for indicator, value in scores.items():
                weight = strategy.weights.get(indicator.lower().replace(' ', '_'), 0.5)
                bar_width = min(weight * 100, 100)
                
                st.markdown(f"""
                <div class="prob-item">
                    <span style="color: #ccc;">{indicator}</span>
                    <span style="color: #00d4ff; font-weight: bold;">{value}</span>
                </div>
                <div class="prob-bar" style="width: {bar_width}%;"></div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)


def render_signal_analysis(strategy, df):
    """Render detailed signal analysis expander"""
    if df is None or len(df) < 51:
        return
    
    row = df.iloc[-1]
    filter_status = strategy.get_filter_status(df)
    spike_status = strategy.get_spike_status(df)
    probability, scores = strategy.calculate_buy_probability(df, len(df)-1)
    
    with st.expander("🔍 Signal Analysis - Why This Signal?", expanded=False):
        current_signal, _, _ = strategy.get_current_signal(df)
        
        st.markdown(f"### Current Signal: **{current_signal}**")
        
        st.markdown("#### Filter Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Required Filters:**")
            
            has_spike = filter_status.get('has_spike', False)
            st.markdown(f"- Spike Detection: {'✅ PASS' if has_spike else '❌ FAIL'} - {'At least 1 spike active' if has_spike else 'No spikes detected'}")
            
            macd_pass = filter_status.get('macd_gate', False)
            macd_hist = row['MACD_Hist']
            macd_roc = row['MACD_Hist_ROC']
            threshold = strategy.settings['macdHistogramRocThreshold'] / 100.0
            st.markdown(f"- MACD Gate: {'✅ PASS' if macd_pass else '❌ FAIL'}")
            st.caption(f"  Hist: {macd_hist:.4f} (need ≤0), ROC: {macd_roc:.4f} (need ≥{threshold:.4f})")
            
            stoch_pass = filter_status.get('stochastic', False)
            stoch_val = row['Stoch_K']
            st.markdown(f"- Stochastic: {'✅ PASS' if stoch_pass else '❌ FAIL'}")
            st.caption(f"  K: {stoch_val:.1f} (need <{strategy.settings['stochasticOversoldThreshold']})")
            
        with col2:
            rvol_pass = filter_status.get('rvol', False)
            rvol_val = row['RVOL']
            st.markdown(f"- RVOL: {'✅ PASS' if rvol_pass else '❌ FAIL'}")
            st.caption(f"  Current: {rvol_val:.2f}x (need >{strategy.settings['rvolThreshold']}x)")
            
            ema_pass = filter_status.get('ema_trend', False)
            st.markdown(f"- EMA Trend: {'✅ PASS' if ema_pass else '❌ FAIL'}")
            st.caption(f"  Price: ${row['Close']:.2f}, EMA9: ${row['EMA_9']:.2f}, EMA20: ${row['EMA_20']:.2f}")
            
            momentum_pass = filter_status.get('bullish_momentum', False)
            st.markdown(f"- Bullish Momentum: {'✅ PASS' if momentum_pass else '❌ FAIL'}")
            st.caption(f"  Price ROC: {row['Price_ROC']:.2f}%, Vol ROC: {row['Volume_ROC']:.2f}%")
        
        st.markdown("---")
        st.markdown("#### Probability Score")
        st.progress(probability)
        st.markdown(f"**{probability*100:.1f}%** (threshold: {strategy.settings['comboSignalThreshold']*100:.0f}%)")
        
        if probability < strategy.settings['comboSignalThreshold']:
            st.warning(f"Probability is below threshold by {(strategy.settings['comboSignalThreshold'] - probability)*100:.1f}%")
        else:
            st.success("Probability meets threshold requirement")


def main():
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.image("logo2.jpg", width=80)
    with col_title:
        st.markdown('<h1 class="main-header">📈 Spiketrade</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Real-time buy/sell signals with 1-minute data (includes pre/post market)</p>', unsafe_allow_html=True)
    
    strategy = PennyBreakoutStrategy()
    
    render_sidebar(strategy)
    
    if 'selected_ticker' not in st.session_state:
        st.session_state['selected_ticker'] = ""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        default_value = st.session_state.get('selected_ticker', '')
        ticker = st.text_input("Enter Stock Ticker", value=default_value, placeholder="e.g., AAPL, TSLA, GME").upper()
        if ticker:
            st.session_state['selected_ticker'] = ticker
    
    with col2:
        auto_refresh = st.checkbox("Auto-Refresh (15s)", value=False)
    
    if ticker:
        with st.spinner(f"Fetching 1-minute data for {ticker}..."):
            df, error = fetch_stock_data(ticker)
        
        if error:
            st.error(f"Error: {error}")
        elif df is not None and not df.empty:
            df = strategy.calculate_signals(df)
            trades = strategy.get_paired_trades(df)
            
            current_signal, signal_reason, probability = strategy.get_current_signal(df)
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = df['Close'].iloc[-1]
                open_price = df['Open'].iloc[0]
                day_change = ((current_price - open_price) / open_price * 100)
                st.metric("Current Price", f"${current_price:.2f}", f"{day_change:+.2f}% today")
            
            with col2:
                prob_display = probability * 100
                if prob_display >= 70:
                    prob_class = "probability-high"
                elif prob_display >= 40:
                    prob_class = "probability-medium"
                else:
                    prob_class = "probability-low"
                st.markdown("**Buy Probability** ℹ️")
                st.markdown(f'<p class="{prob_class}">{prob_display:.1f}%</p>', unsafe_allow_html=True)
                with st.expander("What does this mean?", expanded=False):
                    st.markdown("""
                    The **Buy Probability** represents the likelihood of a profitable trade based on:
                    - Historical backtesting results from similar market conditions
                    - Calibrated weights of technical indicators (spikes, momentum, volume)
                    - Confirmed signal filters (MACD, Stochastic, RVOL, EMA trend)
                    
                    **Higher % = More confidence** that current conditions match profitable setups. This is your estimated edge for entering a position at this moment.
                    """)
            
            with col3:
                if current_signal == "BUY":
                    st.markdown(f'<div class="buy-signal">🟢 {current_signal}</div>', unsafe_allow_html=True)
                elif current_signal == "SELL":
                    st.markdown(f'<div class="sell-signal">🔴 {current_signal}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="hold-signal">🟡 {current_signal}</div>', unsafe_allow_html=True)
            
            with col4:
                rsi_val = df['RSI'].iloc[-1]
                rsi_color = "#00ff88" if rsi_val < 30 else "#ff4757" if rsi_val > 70 else "#ffc107"
                st.markdown(f"""
                <div class="summary-stat">
                    <div class="stat-value" style="color: {rsi_color};">{rsi_val:.1f}</div>
                    <div class="stat-label">RSI</div>
                </div>
                """, unsafe_allow_html=True)
            
            render_indicator_dashboard(strategy, df)
            
            render_probability_breakdown(strategy, df)
            
            render_signal_analysis(strategy, df)
            
            st.plotly_chart(create_chart(df, ticker, trades), use_container_width=True)
            
            st.markdown("---")
            st.subheader("📈 Current Indicators")
            
            latest = df.iloc[-1]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("EMA 9", f"${latest['EMA_9']:.2f}")
                st.metric("EMA 20", f"${latest['EMA_20']:.2f}")
            
            with col2:
                st.metric("MACD", f"{latest['MACD']:.4f}")
                st.metric("MACD Signal", f"{latest['MACD_Signal']:.4f}")
            
            with col3:
                st.metric("Stoch K", f"{latest['Stoch_K']:.1f}")
                st.metric("Stoch D", f"{latest['Stoch_D']:.1f}")
            
            with col4:
                st.metric("VWAP", f"${latest['VWAP']:.2f}")
                st.metric("RVOL", f"{latest['RVOL']:.2f}x")
            
            with col5:
                st.metric("MFI", f"{latest['MFI']:.1f}")
                st.metric("Price ROC", f"{latest['Price_ROC']:.2f}%")
            
            st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            
            if auto_refresh:
                import time
                time.sleep(15)
                st.rerun()
    else:
        st.info("👆 Enter a stock ticker above to start analyzing")
        
        st.markdown("---")
        st.markdown("### How the Penny Breakout Strategy Works")
        st.markdown("""
        <div class="card">
        This strategy identifies breakout opportunities using calibrated technical indicators:
        
        - **Spike Detection**: Z-score analysis on Price, RSI, OBV, MFI, Volume, %B, and VWAP ROC
        - **Probability Calculation**: Weighted scoring using historically calibrated win rates
        - **Signal Filters**: MACD histogram ROC, Stochastic oversold, RVOL threshold, EMA trend
        - **Risk Management**: 2% stop loss, 2% profit target
        - **One Trade Max**: Only one position open at a time
        
        **Chart shows 1-day of 1-minute data including pre/post market hours.**
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Popular Tickers")
        popular = ["AAPL", "TSLA", "GME", "AMC", "NVDA", "AMD"]
        cols = st.columns(6)
        for i, tick in enumerate(popular):
            with cols[i]:
                if st.button(tick, key=f"pop_{tick}", use_container_width=True):
                    st.session_state['selected_ticker'] = tick
                    st.rerun()


if __name__ == "__main__":
    main()
