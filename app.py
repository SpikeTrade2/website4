import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import pytz
from functools import lru_cache

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================

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
    
    .indicator-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 3px;
    }
    
    .badge-on { background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%); color: #1a1a1a; }
    .badge-off { background: #333; color: #666; }
    
    .market-open { background: linear-gradient(135deg, #00ff88 0%, #28a745 100%); color: white; padding: 10px 20px; border-radius: 10px; font-weight: bold; text-align: center; }
    .market-prepost { background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%); color: #1a1a1a; padding: 10px 20px; border-radius: 10px; font-weight: bold; text-align: center; }
    .market-closed { background: linear-gradient(135deg, #dc3545 0%, #c0392b 100%); color: white; padding: 10px 20px; border-radius: 10px; font-weight: bold; text-align: center; }
    
    .sidebar-card { background: linear-gradient(145deg, #1e1e2e 0%, #252538 100%); border: 1px solid #333; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .sidebar-title { font-size: 1rem; font-weight: bold; color: #00d4ff; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 8px; }
    .summary-stat { background: linear-gradient(145deg, #252538 0%, #1e1e2e 100%); border: 1px solid #333; border-radius: 10px; padding: 15px; text-align: center; }
    .stat-value { font-size: 1.8rem; font-weight: bold; color: #00d4ff; }
    .stat-label { font-size: 0.9rem; color: #888; margin-top: 5px; }
    .legend-item { display: flex; align-items: center; gap: 10px; margin: 8px 0; color: #ccc; }
    .legend-color { width: 20px; height: 20px; border-radius: 4px; }
    .prob-bar { height: 6px; background: linear-gradient(90deg, #00d4ff 0%, #00ff88 100%); border-radius: 3px; margin-top: 4px; }
    .prob-item { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: rgba(255, 255, 255, 0.05); border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SETTINGS & CONSTANTS
# ==========================================

CALIBRATED_WEIGHTS = {
    "price_roc": 0.44, "vwap": 0.25, "volume_spike": 0.49,
    "rsi_oversold": 0.85, "rvol_high": 0.44, "obv_roc": 0.51,
    "mfi": 0.67, "spike_quality": 0.48, "ema_downtrend": 0.46,
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

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================

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

# ==========================================
# 4. STRATEGY CLASS (OPTIMIZED)
# ==========================================

class PennyBreakoutStrategy:
    def __init__(self):
        self.settings = TRADING_SETTINGS
        self.weights = CALIBRATED_WEIGHTS
    
    @lru_cache(maxsize=32)
    def calculate_rsi(self, prices, period=14):
        # Converted to static-like method via caching, but passed as instance method
        # Note: In production, better to move outside class or use staticmethod
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).fillna(0).rolling(window=period).mean().replace(0, 1e-10)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_mfi(self, df, period=14):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        # Vectorized implementation
        shifted_tp = typical_price.shift(1)
        positive_flow = money_flow.where(typical_price > shifted_tp, 0)
        negative_flow = money_flow.where(typical_price < shifted_tp, 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum().replace(0, 1e-10)
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def calculate_obv(self, df):
        # Vectorized implementation
        obv_change = pd.Series(0.0, index=df.index)
        shifted_close = df['Close'].shift(1)
        
        # Determine direction
        direction = np.select(
            [df['Close'] > shifted_close, df['Close'] < shifted_close],
            [1, -1],
            default=0
        )
        
        # Standard OBV usually ignores flat volume, but for penny stocks
        # we often treat flat consolidation as continuation or neutral.
        # Strict OBV:
        obv = (direction * df['Volume']).cumsum()
        return obv
    
    def calculate_roc(self, series, period):
        return series.pct_change(period).fillna(0) * 100
    
    def calculate_z_score(self, series, lookback=20):
        mean = series.rolling(window=lookback).mean()
        std = series.rolling(window=lookback).std().replace(0, 1e-10)
        return (series - mean) / std
    
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
        # Avoid division by zero
        denom = (upper - lower).replace(0, 1e-10)
        percent_b = (prices - lower) / denom
        return upper, middle, lower, percent_b
    
    def calculate_stochastic(self, df, k_period=14, d_period=3):
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        denom = (high_max - low_min).replace(0, 1e-10)
        stoch_k = 100 * (df['Close'] - low_min) / denom
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d
    
    def calculate_vwap(self, df):
        # Optimized: Reset VWAP daily
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vp = typical_price * df['Volume']
        
        # Group by date to reset calculation every day
        # This is CRITICAL for accurate intraday VWAP
        dates = df.index.date
        vwap_num = vp.groupby(dates).cumsum()
        vwap_denom = df['Volume'].groupby(dates).cumsum().replace(0, 1e-10)
        
        return vwap_num / vwap_denom
    
    def calculate_rvol(self, df, period=20):
        avg_volume = df['Volume'].rolling(window=period).mean().replace(0, 1e-10)
        rvol = df['Volume'] / avg_volume
        return rvol
    
    def calculate_signals(self, df):
        df = df.copy()
        
        # Technical Indicators
        # Passing self.calculate_rsi to bypass scope issue with lru_cache if needed
        # but here we call the method normally.
        df['RSI'] = self.calculate_rsi(df['Close'], self.settings['rsiLengthMinutes'])
        df['MFI'] = self.calculate_mfi(df, self.settings['mfiPeriodMinutes'])
        df['OBV'] = self.calculate_obv(df)
        
        # ROCs
        df['Price_ROC'] = self.calculate_roc(df['Close'], self.settings['priceRocPeriodMinutes'])
        df['Volume_ROC'] = self.calculate_roc(df['Volume'], self.settings['priceRocPeriodMinutes'])
        df['RSI_ROC'] = self.calculate_roc(df['RSI'], self.settings['rsiLengthMinutes'])
        df['OBV_ROC'] = self.calculate_roc(df['OBV'], self.settings['obvRocPeriodMinutes'])
        df['MFI_ROC'] = self.calculate_roc(df['MFI'], self.settings['mfiPeriodMinutes'])
        
        # Z-Scores
        df['Price_ROC_Z'] = self.calculate_z_score(df['Price_ROC'])
        df['Volume_ROC_Z'] = self.calculate_z_score(df['Volume_ROC'])
        df['RSI_ROC_Z'] = self.calculate_z_score(df['RSI_ROC'])
        df['OBV_ROC_Z'] = self.calculate_z_score(df['OBV_ROC'])
        df['MFI_ROC_Z'] = self.calculate_z_score(df['MFI_ROC'])
        
        # EMAs & MACD
        df['EMA_9'] = self.calculate_ema(df['Close'], 9)
        df['EMA_20'] = self.calculate_ema(df['Close'], 20)
        df['EMA_50'] = self.calculate_ema(df['Close'], 50)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(df['Close'])
        df['MACD_Hist_ROC'] = df['MACD_Hist'].pct_change() # Using pct_change is cleaner
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'], df['Percent_B'] = self.calculate_bollinger_bands(df['Close'], self.settings['bbLengthMinutes'])
        df['Percent_B_ROC'] = self.calculate_roc(df['Percent_B'], self.settings['priceRocPeriodMinutes'])
        df['Percent_B_ROC_Z'] = self.calculate_z_score(df['Percent_B_ROC'])
        
        # Stochastic & VWAP
        df['Stoch_K'], df['Stoch_D'] = self.calculate_stochastic(df)
        df['VWAP'] = self.calculate_vwap(df)
        df['VWAP_ROC'] = self.calculate_roc(df['VWAP'], self.settings['vwapPeriodMinutes'])
        df['VWAP_ROC_Z'] = self.calculate_z_score(df['VWAP_ROC'])
        
        # RVOL & Volume Spikes
        df['RVOL'] = self.calculate_rvol(df)
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Spike'] = df['Volume'] > (df['Volume_MA'] * 1.5)
        
        # Spike Flags (Boolean)
        df['Price_Spike'] = df['Price_ROC_Z'] > self.settings['spikePriceRocZThreshold']
        df['RSI_Spike'] = df['RSI_ROC_Z'] > self.settings['spikeRsiRocZThreshold']
        df['OBV_Spike'] = df['OBV_ROC_Z'] > self.settings['spikeObvRocZThreshold']
        df['MFI_Spike'] = df['MFI_ROC_Z'] > self.settings['spikeMfiRocZThreshold']
        df['Vol_ROC_Spike'] = df['Volume_ROC_Z'] > self.settings['spikeVolumeRocZThreshold']
        df['Percent_B_Spike'] = df['Percent_B_ROC_Z'] > self.settings['spikePercentBRocZThreshold']
        df['VWAP_Spike'] = df['VWAP_ROC_Z'] > self.settings['spikeVwapRocZThreshold']
        
        df = self.generate_signals_optimized(df)
        
        return df
    
    def generate_signals_optimized(self, df):
        df['Buy_Signal'] = False
        df['Sell_Signal'] = False
        df['Signal_Probability'] = 0.0
        
        # 1. Vectorized Pre-calculation of conditions
        # -------------------------------------------
        
        # Condition A: Spikes
        has_spike = (df['Price_Spike'] | df['RSI_Spike'] | 
                     df['OBV_Spike'] | df['MFI_Spike'] | df['Vol_ROC_Spike'] |
                     df['Percent_B_Spike'] | df['VWAP_Spike'])
        
        # Condition B: Bullish Momentum
        bullish_momentum = (df['Price_ROC'] > 0) & (df['Volume_ROC'] > 0) & (df['OBV_ROC'] > 0)
        
        # Condition C: MACD Gate
        macd_threshold = self.settings['macdHistogramRocThreshold'] / 100.0 # Normalize if needed
        # Note: Original code used diff() / price, here checking raw threshold or pct_change
        # Adjusting logic to match typical MACD turn up:
        macd_valid = (df['MACD_Hist'] <= 0) & (df['MACD_Hist_ROC'] > 0) 
        
        # Condition D: Oversold
        stoch_oversold = df['Stoch_K'] < self.settings['stochasticOversoldThreshold']
        
        # Condition E: RVOL
        rvol_valid = df['RVOL'] > self.settings['rvolThreshold']
        
        # Condition F: Trend (Price not below all EMAs)
        price_below_all_emas = (df['Close'] < df['EMA_9']) & \
                               (df['Close'] < df['EMA_20']) & \
                               (df['Close'] < df['EMA_50'])
        
        # Master Potential Buy Mask
        potential_buys = (has_spike & bullish_momentum & macd_valid & 
                          stoch_oversold & rvol_valid & (~price_below_all_emas))
        
        # 2. Probability Calculation (Semi-Vectorized)
        # --------------------------------------------
        # To strictly avoid O(N) scoring, we would need to vectorize the scoring.
        # For now, we will compute scores row-by-row ONLY for potential buys + last row (for UI)
        # to save processing time.
        
        # Calculate prob for the very last row (always needed for UI)
        last_idx = df.index[-1]
        prob_last, _ = self.calculate_buy_probability(df, -1)
        df.at[last_idx, 'Signal_Probability'] = prob_last

        # Identify indices where conditions are met
        candidate_indices = np.where(potential_buys)[0]

        # Calculate probabilities only for candidate rows to save time
        # We use integer indexing for speed
        prob_col_idx = df.columns.get_loc('Signal_Probability')
        
        for idx in candidate_indices:
            if idx < 50: continue # Skip warmup period
            
            # Calculate score
            prob, _ = self.calculate_buy_probability(df, idx)
            df.iat[idx, prob_col_idx] = prob
            
        # 3. Trade Simulation (State Machine)
        # -----------------------------------
        # We must iterate linearly to ensure we don't buy while already in a trade
        
        in_trade = False
        entry_price = 0.0
        
        # Get integer locations for faster access
        buy_sig_col = df.columns.get_loc('Buy_Signal')
        sell_sig_col = df.columns.get_loc('Sell_Signal')
        close_col = df.columns.get_loc('Close')
        
        # Pre-convert columns to numpy for speed in the loop
        close_arr = df['Close'].values
        ema9_arr = df['EMA_9'].values
        ema20_arr = df['EMA_20'].values
        ema50_arr = df['EMA_50'].values
        macd_arr = df['MACD'].values
        signal_arr = df['MACD_Signal'].values
        rsi_arr = df['RSI'].values
        price_roc_arr = df['Price_ROC'].values
        vol_spike_arr = df['Volume_Spike'].values
        probs_arr = df['Signal_Probability'].values
        
        # Create a boolean array for the potential buys we identified earlier
        potential_buys_arr = potential_buys.values
        
        for i in range(50, len(df)):
            current_price = close_arr[i]
            
            if not in_trade:
                # Check if this row was identified as a potential buy AND meets probability threshold
                if potential_buys_arr[i] and probs_arr[i] >= self.settings['comboSignalThreshold']:
                    df.iat[i, buy_sig_col] = True
                    in_trade = True
                    entry_price = current_price
            
            else:
                # We are in a trade, check exit conditions
                pnl_pct = (current_price - entry_price) / entry_price
                
                hit_stop_loss = pnl_pct <= -self.settings['stopLossPct']
                hit_target = pnl_pct >= self.settings['targetGainPercent'] / 100
                
                # Technical exits
                macd_bearish = (macd_arr[i] < signal_arr[i]) and (macd_arr[i-1] >= signal_arr[i-1])
                rsi_overbought = rsi_arr[i] > 70
                
                below_all_emas = (current_price < ema9_arr[i]) and \
                                 (current_price < ema20_arr[i]) and \
                                 (current_price < ema50_arr[i])
                
                bearish_momentum = (price_roc_arr[i] < -1) and vol_spike_arr[i]
                
                if hit_stop_loss or hit_target or macd_bearish or (rsi_overbought and bearish_momentum) or below_all_emas:
                    df.iat[i, sell_sig_col] = True
                    in_trade = False
                    entry_price = 0.0
        
        return df
    
    def calculate_buy_probability(self, df, idx):
        # Handle negative indexing or too small index
        if idx < 0: idx = len(df) + idx
        if idx < 50: return 0.0, {}
        
        # We use .iloc specifically for this row
        row = df.iloc[idx]
        
        scores = {}
        total_weight = 0.0
        weighted_score = 0.0
        
        # Helper to add score
        def add_score(name, condition, weight_key, score_val=1.0, fmt_val=""):
            nonlocal total_weight, weighted_score
            w = self.weights.get(weight_key, 0.5)
            if condition:
                weighted_score += score_val * w
                total_weight += w
                scores[name] = fmt_val
            return w # return weight if needed for partials
            
        # 1. Price ROC
        if row['Price_ROC'] > 0:
            s = min(1.0, row['Price_ROC'] / self.settings['regularPriceRocThreshold'])
            w = self.weights['price_roc']
            weighted_score += s * w
            total_weight += w
            scores['Price ROC'] = f"+{s*100:.0f}%"
            
        # 2. Volume Spike
        add_score('Volume Spike', row['Volume_Spike'], 'volume_spike', fmt_val="Active")
        
        # 3. RSI
        if row['RSI'] < 30:
            add_score('RSI Oversold', True, 'rsi_oversold', fmt_val=f"{row['RSI']:.1f}")
        elif row['RSI'] < 40:
            w = self.weights['rsi_oversold'] * 0.5
            weighted_score += w
            total_weight += w
            scores['RSI Low'] = f"{row['RSI']:.1f}"
            
        # 4. RVOL
        if row['RVOL'] > self.settings['rvolThreshold']:
            s = min(1.0, (row['RVOL'] - 1) / 1.0)
            w = self.weights['rvol_high']
            weighted_score += s * w
            total_weight += w
            scores['RVOL High'] = f"{row['RVOL']:.2f}x"
            
        # 5. OBV ROC
        if row['OBV_ROC'] > self.settings['regularObvRocThreshold']:
            s = min(1.0, row['OBV_ROC'] / (self.settings['regularObvRocThreshold'] * 2))
            w = self.weights['obv_roc']
            weighted_score += s * w
            total_weight += w
            scores['OBV ROC'] = f"+{row['OBV_ROC']:.1f}%"
            
        # 6. MFI
        if row['MFI'] < 30:
            add_score('MFI Oversold', True, 'mfi', fmt_val=f"{row['MFI']:.1f}")
        
        # 7. Stoch
        if row['Stoch_K'] < self.settings['stochasticOversoldThreshold']:
            s = 1.0 - (row['Stoch_K'] / self.settings['stochasticOversoldThreshold'])
            w = self.weights['stoch_oversold']
            weighted_score += s * w
            total_weight += w
            scores['Stoch Oversold'] = f"{row['Stoch_K']:.1f}"
            
        # 8. EMAs
        if row['Close'] > row['EMA_9'] and row['Close'] > row['EMA_20']:
            add_score('Above EMAs', True, 'ema_downtrend', fmt_val="Yes")
            
        # 9. Spike Quality (Multi-spike)
        spike_count = sum([row['Price_Spike'], row['RSI_Spike'], row['OBV_Spike'], 
                          row['MFI_Spike'], row['Vol_ROC_Spike'], row['Percent_B_Spike'],
                          row['VWAP_Spike']])
        if spike_count >= 2:
            s = min(1.0, spike_count / 4.0)
            w = self.weights['spike_quality']
            weighted_score += s * w
            total_weight += w
            scores['Multi-Spike'] = f"{spike_count} spikes"
            
        # 10. VWAP
        if row['Close'] > row['VWAP']:
            add_score('Above VWAP', True, 'vwap', fmt_val="Yes")

        # Final calc
        if total_weight > 0:
            probability = weighted_score / total_weight
        else:
            probability = 0.0
            
        return max(0.0, min(1.0, probability)), scores

    def get_current_signal(self, df):
        if len(df) < 2: return "HOLD", "Insufficient data", 0.0
        latest = df.iloc[-1]
        
        if latest['Buy_Signal']:
            return "BUY", "Buy signal triggered", latest['Signal_Probability']
        elif latest['Sell_Signal']:
            return "SELL", "Sell signal triggered", 0.0
        
        return "HOLD", "Monitoring", latest['Signal_Probability']
    
    def get_filter_status(self, df):
        if len(df) < 51: return {}
        row = df.iloc[-1]
        
        # Re-eval for UI
        has_spike = (row['Price_Spike'] or row['RSI_Spike'] or row['OBV_Spike'] or 
                     row['MFI_Spike'] or row['Vol_ROC_Spike'] or row['Percent_B_Spike'] or row['VWAP_Spike'])
        
        bullish_momentum = row['Price_ROC'] > 0 and row['Volume_ROC'] > 0 and row['OBV_ROC'] > 0
        macd_threshold = self.settings['macdHistogramRocThreshold'] / 100.0
        macd_valid = row['MACD_Hist'] <= 0 and row['MACD_Hist_ROC'] > 0 # Simplified check
        
        return {
            'has_spike': bool(has_spike),
            'bullish_momentum': bool(bullish_momentum),
            'macd_gate': bool(macd_valid),
            'stochastic': bool(row['Stoch_K'] < self.settings['stochasticOversoldThreshold']),
            'rvol': bool(row['RVOL'] > self.settings['rvolThreshold']),
            'ema_trend': not (row['Close'] < row['EMA_9'] and row['Close'] < row['EMA_20'])
        }

    def get_spike_status(self, df):
        if len(df) < 51: return {}
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
        buy_indices = df.index[df['Buy_Signal']]
        sell_indices = df.index[df['Sell_Signal']]
        
        for buy_idx in buy_indices:
            # Find next sell signal after this buy
            valid_sells = sell_indices[sell_indices > buy_idx]
            
            entry_price = df.loc[buy_idx]['Close']
            prob = df.loc[buy_idx]['Signal_Probability']
            
            if not valid_sells.empty:
                sell_idx = valid_sells[0]
                exit_price = df.loc[sell_idx]['Close']
                pnl = ((exit_price - entry_price) / entry_price) * 100
                
                trades.append({
                    'entry_time': buy_idx, 'exit_time': sell_idx,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'pnl_pct': pnl, 'probability': prob, 'open': False
                })
            else:
                # Open trade
                trades.append({
                    'entry_time': buy_idx, 'exit_time': None,
                    'entry_price': entry_price, 'exit_price': None,
                    'pnl_pct': None, 'probability': prob, 'open': True
                })
        return trades

# ==========================================
# 5. DATA FETCHING & PLOTTING
# ==========================================

def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Fetch 5 days to ensure we have enough EMA/MACD warmup data
        df = stock.history(period="5d", interval="1m", prepost=True)
        
        if df.empty:
            return None, "No data available."
            
        # Filter to keep only today's data for the chart, but keep history for calc?
        # Actually, for 1m interval, 5d is a lot. Let's do 2d.
        
        # Check if we have recent data
        last_dt = df.index[-1].date()
        today = datetime.now().date()
        
        # If the last data is not from today (and it's a weekday), warn?
        # We'll just return what we have.
        
        return df, None
    except Exception as e:
        return None, str(e)

def create_chart(df, ticker, trades):
    # Slice to last 1 day for visualization to prevent lag
    # Find start of 'today' or last available day
    last_day = df.index[-1].normalize()
    plot_df = df[df.index >= last_day]
    
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.4, 0.1, 0.15, 0.15, 0.2],
        subplot_titles=(f'{ticker}', 'Signal Prob', 'Volume', 'RSI/Stoch', 'MACD')
    )
    
    # Price & EMAs
    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
        low=plot_df['Low'], close=plot_df['Close'], name='Price'
    ), row=1, col=1)
    
    for ema, color in [('EMA_9', '#FF9800'), ('EMA_20', '#9C27B0'), ('EMA_50', '#2196F3')]:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[ema], name=ema, line=dict(color=color, width=1)), row=1, col=1)
        
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['VWAP'], name='VWAP', line=dict(color='#00BCD4', width=1.5, dash='dot')), row=1, col=1)

    # Trades
    for trade in trades:
        # Only plot if within view
        if trade['entry_time'] < plot_df.index[0]: continue
        
        fig.add_trace(go.Scatter(
            x=[trade['entry_time']], y=[trade['entry_price']*0.998], mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='#00ff88'),
            name='Buy', showlegend=False
        ), row=1, col=1)
        
        if not trade['open'] and trade['exit_time'] >= plot_df.index[0]:
            fig.add_trace(go.Scatter(
                x=[trade['exit_time']], y=[trade['exit_price']*1.002], mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='#ff4757'),
                name='Sell', showlegend=False
            ), row=1, col=1)

    # Probability
    prob_vals = plot_df['Signal_Probability'] * 100
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=prob_vals, name='Prob %',
        line=dict(color='#00d4ff', width=1), fill='tozeroy', fillcolor='rgba(0,212,255,0.1)'
    ), row=2, col=1)
    fig.add_hline(y=76, line_dash="dash", line_color="#00ff88", row=2, col=1)
    
    # Volume
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(plot_df['Close'], plot_df['Open'])]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=colors, name='Vol'), row=3, col=1)
    
    # RSI & Stoch
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], name='RSI', line=dict(color='#2196F3')), row=4, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Stoch_K'], name='Stoch', line=dict(color='#FF9800', width=1)), row=4, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="white", row=4, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="white", row=4, col=1)
    
    # MACD
    colors_macd = ['#26a69a' if v >= 0 else '#ef5350' for v in plot_df['MACD_Hist']]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['MACD_Hist'], marker_color=colors_macd, name='Hist'), row=5, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD'], line=dict(color='#2196F3', width=1), name='MACD'), row=5, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD_Signal'], line=dict(color='#FF9800', width=1), name='Sig'), row=5, col=1)

    fig.update_layout(
        height=900, template='plotly_dark',
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1, x=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,10,10,0.5)'
    )
    return fig

# ==========================================
# 6. UI COMPONENT FUNCTIONS
# ==========================================

def safe_image_load():
    """Safely loads logo or returns None to prevent crashes"""
    if os.path.exists("logo2.jpg"):
        return "logo2.jpg"
    return None 

def render_sidebar(strategy):
    with st.sidebar:
        logo = safe_image_load()
        if logo:
            st.image(logo, width=120)
        else:
            st.markdown("### 📈 Spiketrade")
            
        st.markdown("---")
        
        status, text = get_market_status()
        color = "#00ff88" if status == "open" else "#ffc107" if status == "prepost" else "#ff4757"
        st.markdown(f"Market Status: <span style='color:{color}; font-weight:bold'>{text}</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("Settings")
        st.markdown(f"**Threshold**: {strategy.settings['comboSignalThreshold']*100:.0f}%")
        st.markdown(f"**Stop Loss**: {strategy.settings['stopLossPct']*100:.0f}%")

def render_indicator_dashboard(strategy, df):
    if df is None or len(df) < 51: return
    
    spike_status = strategy.get_spike_status(df)
    filter_status = strategy.get_filter_status(df)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Detectors**")
        html = ""
        for k, v in spike_status.items():
            cls = "badge-on" if v else "badge-off"
            html += f"<span class='indicator-badge {cls}'>{k}</span>"
        st.markdown(html, unsafe_allow_html=True)
        
    with c2:
        st.markdown("**Filters**")
        html = ""
        for k, v in filter_status.items():
            cls = "filter-pass" if v else "filter-fail"
            html += f"<span class='{cls}'>{k}</span>"
        st.markdown(html, unsafe_allow_html=True)

def render_prob_breakdown(strategy, df):
    if df is None: return
    prob, scores = strategy.calculate_buy_probability(df, len(df)-1)
    if not scores: return
    
    with st.expander("📊 Probability Breakdown"):
        for k, v in scores.items():
            st.markdown(f"""
            <div class='prob-item'>
                <span>{k}</span> <span style='color:#00d4ff'>{v}</span>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 7. MAIN APP LOOP
# ==========================================

def main():
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        logo = safe_image_load()
        if logo:
            st.image(logo, width=80)
    with col_title:
        st.markdown('<h1 class="main-header">📈 Spiketrade Optimized</h1>', unsafe_allow_html=True)

    strategy = PennyBreakoutStrategy()
    render_sidebar(strategy)
    
    if 'selected_ticker' not in st.session_state:
        st.session_state['selected_ticker'] = "SPY"
        
    c1, c2 = st.columns([3, 1])
    with c1:
        ticker = st.text_input("Ticker", value=st.session_state['selected_ticker']).upper()
        st.session_state['selected_ticker'] = ticker
    with c2:
        auto_refresh = st.checkbox("Auto-Refresh (15s)")
        
    if ticker:
        with st.spinner("Analyzing..."):
            df, err = fetch_stock_data(ticker)
            
        if err:
            st.error(err)
        elif df is not None:
            # Process Signals
            df = strategy.calculate_signals(df)
            trades = strategy.get_paired_trades(df)
            signal, reason, prob = strategy.get_current_signal(df)
            
            # Metrics
            latest = df.iloc[-1]
            m1, m2, m3 = st.columns(3)
            m1.metric("Price", f"${latest['Close']:.2f}")
            m2.metric("Probability", f"{prob*100:.1f}%")
            
            sig_html = f"<div class='{signal.lower()}-signal'>{signal}</div>"
            if signal == "BUY": sig_html = f"<div class='buy-signal'>BUY</div>"
            elif signal == "SELL": sig_html = f"<div class='sell-signal'>SELL</div>"
            else: sig_html = f"<div class='hold-signal'>HOLD</div>"
            
            m3.markdown(sig_html, unsafe_allow_html=True)
            
            # Visuals
            render_indicator_dashboard(strategy, df)
            render_prob_breakdown(strategy, df)
            
            st.plotly_chart(create_chart(df, ticker, trades), use_container_width=True)
            
            if auto_refresh:
                import time
                time.sleep(15)
                st.rerun()

if __name__ == "__main__":
    main()
