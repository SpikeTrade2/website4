# SpikeTrade - Replit.md

### Overview
SpikeTrade is a comprehensive stock trading analysis platform. This Replit environment hosts a **Streamlit web application** that replicates the penny breakout trading strategy from the original Java desktop application. The web app accepts ticker symbols and provides probability-based trading signals, recommendations, and price projections.

### User Preferences
As an AI agent, focus on essential information needed for coding tasks. Prioritize architectural decisions and high-level feature specifications. Avoid changes to `.cache`, `.local`, and `attached_assets` directories. Explain the "why" behind code changes. Inform before major architectural shifts.

### Streamlit Web Application (November 27, 2025)
The primary component is `app.py` - a full-featured Streamlit dashboard that provides:

**Features:**
- Real-time stock analysis using Yahoo Finance data
- **1-Minute candlestick charts** with EMA overlays and volume
- Probability-based BUY/SELL/HOLD recommendations
- Technical indicators: RSI, MACD, MFI, OBV, Bollinger Bands, Stochastic, VWAP, EMA, RVOL
- Spike detection using z-score thresholds
- Profit/loss projections with stop-loss and target prices
- Predicted time to profit based on momentum analysis
- Market status indicator (open/closed/pre-market/after-hours)
- Professional dark theme with SpikeTrade logo

**Technical Implementation:**
- `ProbabilityCalibrator`: Logistic regression using calibrated weights from `indicator_calibration.json`
- `PredictionEngine`: ROC-based time estimation with volatility adjustment
- `TechnicalIndicators`: Full suite of indicators matching Java implementation
- `PennyBreakoutAnalyzer`: Main analysis engine that calculates all indicators and generates signals

**Configuration Files:**
- `trading_settings.json`: Thresholds, periods, and trading parameters
- `indicator_calibration.json`: Calibrated weights for probability calculation
- `.streamlit/config.toml`: Streamlit server and theme configuration

### System Architecture (Java Desktop App)
The original SpikeTrade desktop application uses:
- **UI/UX:** JavaFX with real-time updates and traffic light signal indicators
- **Backend:** gRPC services for market data streaming and trade execution
- **Analysis:** `SpikeAnalyzer` for technical analysis, `PredictionEngine` for probability calibration
- **Risk Management:** Automated stop losses, trailing stops, position limiting

### File Structure
```
/app.py                     - Main Streamlit web application
/requirements.txt           - Python dependencies
/.streamlit/config.toml     - Streamlit configuration
/static/logo2.jpg          - SpikeTrade logo
/trading_settings.json      - Trading configuration
/indicator_calibration.json - Calibrated indicator weights
/src/main/java/            - Original Java source code (reference)
/build.gradle              - Java build configuration
```

### Replit Environment Setup
- **Language:** Python (Streamlit)
- **Workflow:** "SpikeTrade Dashboard" runs `streamlit run app.py` on port 5000
- **Data Source:** Yahoo Finance (yfinance)
- **Chart Timeframe:** 1-minute intervals for real-time analysis

### Deployment
The Streamlit app is configured for deployment on Streamlit Cloud. The `requirements.txt` contains all necessary dependencies. For Streamlit Cloud deployment, connect the GitHub repository and set the main file path to `app.py`.
