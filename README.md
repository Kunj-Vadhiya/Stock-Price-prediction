# Stock Price Prediction 

**Required Packages:**
- `yfinance` - Yahoo Finance data download
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scikit-learn` - Data preprocessing and metrics
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `plotly` - Interactive charts
- `ta` - Technical analysis indicators
- `xgboost` - Gradient boosting classifier
- `tensorflow` - Deep learning framework
- `keras` - Neural network API
- `joblib` - Model serialization

### 1️. Data Loading
- Download stock data from Yahoo Finance using `yf.download()`
- Default: Amazon (AMZN) from 2010-01-01 to 2024-01-01
- Extract OHLCV (Open, High, Low, Close, Volume) data
- Handle MultiIndex columns and timezone information

### 2️. Data Preprocessing
- Check and handle missing values using forward/backward fill
- Ensure proper data types and date sorting
- Remove any remaining NaN rows
- Final dataset: **3,522 rows** of clean data

### 3️. Feature Engineering
Create **52 technical indicators and features:**
- **Moving Averages**: MA7, MA21, MA50, MA200
- **Exponential Moving Average**: EMA12
- **RSI**: Relative Strength Index (14-day)
- **MACD**: MACD line, Signal line, Histogram
- **Bollinger Bands**: Upper, Lower, Middle bands
- **Volatility**: 20-day rolling standard deviation
- **Daily Returns**: Percentage change
- **Lag Features**: Close_1 to Close_30 (previous 30 days)
- **Target Variables**: Target_LSTM (next day price), Target_XGB (up/down movement)

After feature engineering: **3,322 usable samples**

### 4️. Data Visualization
Generate interactive charts:
- Price history with volume (candlestick chart)
- Moving averages overlay
- RSI indicator with overbought/oversold zones
- MACD with signal line and histogram
- Bollinger Bands with price action
- Correlation heatmap of all features

### 5️. LSTM Model Training (Price Prediction)

**Data Preparation:**
- Extract Close prices and normalize using MinMaxScaler (0-1 range)
- Create sequences with 60-day lookback window
- Split: 80% training (2,609 samples), 20% testing (653 samples)

**Model Architecture:**
```
Sequential(
    LSTM(50, return_sequences=True, input_shape=(60, 1))
    Dropout(0.2)
    LSTM(50, return_sequences=False)
    Dropout(0.2)
    Dense(25)
    Dense(1)
)
```
- Total parameters: **31,901**
- Optimizer: Adam
- Loss: Mean Squared Error

**Training:**
- Max epochs: 50
- Batch size: 32
- Validation split: 10%
- Early stopping: patience=10 (stopped at epoch 11)

### 6️. XGBoost Model Training (Movement Classification)

**Data Preparation:**
- Use all 48 engineered features (excluding Date and targets)
- Binary target: 1 = UP, 0 = DOWN
- Split: 80% training (2,657 samples), 20% testing (665 samples)

**Model Configuration:**
```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
```

**Training:**
- Gradient boosting with 100 trees
- Maximum depth: 5
- Learning rate: 0.1

### 7️. Model Evaluation & Prediction
- Evaluate both models on test data
- Calculate performance metrics
- Make next-day predictions
- Compare model agreement

## Results

### LSTM Model (Price Prediction)
**Performance Metrics:**
- **Test RMSE**: $3.90
- **Test MAE**: $3.02
- **Test MAPE**: 2.53%

**Next-Day Prediction:**
- **Current Price**: $153.38 (Dec 28, 2023)
- **Predicted Price**: $151.49
- **Expected Change**: -$1.89 (-1.23%)
- **Signal**: **Bearish**

### XGBoost Model (Movement Classification)
**Performance Metrics:**
- **Accuracy**: 53.48%
- **Precision**: 54.37%
- **Recall**: 53.72%
- **F1-Score**: 54.04%
- **ROC-AUC**: 0.5757

**Next-Day Prediction:**
- **Direction**:  **UP**
- **Confidence**: 50.14%
- **Signal**: **Bullish**

**Top 5 Important Features:**
1. Close_1 (Previous day close) - 10.26%
2. Close_2 (2 days ago close) - 6.23%
3. Close_3 (3 days ago close) - 5.64%
4. EMA_12 - 5.23%
5. RSI - 4.76%

### Combined Analysis
**MIXED SIGNALS** - Models disagree:
- **LSTM**: Predicts bearish movement (price drop of 1.23%)
- **XGBoost**: Predicts bullish movement (UP with 50.14% confidence)