# AAPL Stock Analysis with Machine Learning

My first data science project - a comprehensive Apple (AAPL) stock analysis combining technical analysis, risk metrics, Monte Carlo simulations, and machine learning price predictions.

## About This Project

This project was built to explore the intersection of financial analysis and machine learning. I scraped 5 years of Apple stock data using `yfinance` and implemented multiple analytical approaches to understand stock behavior and predict future prices.

### What I Learned

- **Web scraping** financial data with yfinance API
- **Data preprocessing** with pandas (handling dates, missing values, feature engineering)
- **Technical analysis** indicators used by real traders (RSI, MACD, Moving Averages)
- **Risk management** metrics used in portfolio management (VaR, Sortino Ratio, Drawdown)
- **Monte Carlo simulation** for probabilistic price forecasting
- **Deep Learning** with TensorFlow/Keras (Bidirectional LSTM networks)
- **Ensemble methods** with scikit-learn (Random Forest)
- **Data visualization** with matplotlib

## Features

### 1. Data Collection

Scraped 5 years of historical AAPL data using the `yfinance` library:

```python
import yfinance as yf

aapl = yf.download('AAPL', period='5y')
aapl.to_csv('AAPL_5year.csv')
```

### 2. Feature Engineering

The raw stock data is enhanced with calculated technical indicators:

```python
stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['Close']).rsi()
stock_data['MACD'] = ta.trend.MACD(stock_data['Close']).macd()
stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['MA_200'] = stock_data['Close'].rolling(window=200).mean()
```

### 3. Technical Indicators

| Indicator | What It Measures | How It's Used |
|-----------|------------------|---------------|
| **RSI** (Relative Strength Index) | Momentum on a 0-100 scale | Values <30 suggest oversold, >70 suggest overbought |
| **MACD** (Moving Average Convergence Divergence) | Trend direction and momentum | Positive = bullish momentum, Negative = bearish |
| **50-Day MA** | Short-term price trend | Price above MA = uptrend |
| **200-Day MA** | Long-term price trend | Golden/Death cross signals |

### 4. Risk Analysis

Implemented three key risk metrics used by professional portfolio managers:

| Metric | Result | Interpretation |
|--------|--------|----------------|
| **Value at Risk (95%)** | -2.70% | On 95% of days, daily loss won't exceed 2.70% |
| **Sortino Ratio** | 1.15 | Decent risk-adjusted returns (>1 is generally good) |
| **Max Drawdown** | -31.31% | Largest peak-to-trough decline in the dataset |

### 5. Monte Carlo Simulation

Ran 1,000 simulations over 180 days to generate probabilistic price forecasts:

| Scenario | Predicted Price |
|----------|-----------------|
| Bullish (Upper 95%) | $430.47 |
| Bearish (Lower 95%) | $175.94 |

This gives a confidence interval for where the stock price could realistically move.

### 6. Machine Learning Models

#### Bidirectional LSTM Neural Network
- **Architecture**: 2 Bidirectional LSTM layers with Dropout regularization
- **Why Bidirectional**: Captures patterns from both past and future context in sequences
- **Training**: 80/20 train-test split with early stopping to prevent overfitting

```python
model_lstm = Sequential([
    Bidirectional(LSTM(50, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(50, return_sequences=False)),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])
```

#### Random Forest Regressor
- **Ensemble method** using 100 decision trees
- **Purpose**: Baseline comparison for the LSTM model
- **Advantage**: Less prone to overfitting, interpretable feature importance

### 7. Trading Signal Strategy

Simple rule-based system combining RSI and MACD:

| Condition | Signal |
|-----------|--------|
| RSI < 30 AND MACD > 0 | **BUY** (oversold with bullish momentum) |
| RSI > 70 AND MACD < 0 | **SELL** (overbought with bearish momentum) |
| Otherwise | **HOLD** |

## Results Summary

| Metric | Value |
|--------|-------|
| Latest Closing Price | $245.83 |
| 52-Week High | $259.02 |
| 52-Week Low | $116.36 |
| Price Range | $142.66 (122% spread) |
| Current Signal | HOLD |

## Visualization

The project generates a comparative chart showing:
- Actual stock prices (blue)
- LSTM predictions (red dashed)
- Random Forest predictions (green dashed)

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Data Collection** | yfinance |
| **Data Processing** | pandas, NumPy |
| **Visualization** | matplotlib |
| **Technical Analysis** | ta (Technical Analysis Library) |
| **Machine Learning** | scikit-learn (Random Forest) |
| **Deep Learning** | TensorFlow, Keras (LSTM) |

## Project Structure

```
aapl-stock-analysis/
├── README.md                    # Project documentation
├── aapl_stock_analysis.ipynb    # Main analysis notebook
├── requirements.txt             # Python dependencies
└── .gitignore                   # Git ignore rules
```

## Future Improvements

- [ ] Add more technical indicators (Bollinger Bands, ATR, OBV)
- [ ] Implement sentiment analysis from news/social media
- [ ] Create a Streamlit dashboard for interactive analysis
- [ ] Add backtesting framework to validate trading strategy
- [ ] Compare more ML models (XGBoost, Transformer)
