# Bitcoin Price Prediction and Risk Analysis

## Introduction

Bitcoin is a digital currency that operates without a central authority, meaning it isn’t controlled by banks or governments. Instead, it runs on a decentralized system called blockchain, where transactions are verified by a global network of computers. While Bitcoin has gained popularity as an investment and payment method, its price is highly unpredictable, often changing rapidly due to news, market trends, and investor sentiment.

For traders and investors, this volatility presents both opportunities and risks. Without the right tools, predicting Bitcoin’s price movements can be difficult, leading to poor trading decisions and financial losses. Traditional trading methods often rely on experience and intuition, which may not be effective in such a fast-changing market.

## Project Overview

This project leverages multiple time-series forecasting models and sentiment analysis techniques to forecast Bitcoin prices, assess market risks, and generate actionable trading signals. By integrating traditional forecasting methods with modern deep learning and natural language processing (NLP), the system aims to provide traders with data-driven insights for more informed decision-making.

## Project Objectives

1. **Bitcoin Price Prediction**: Predict short-term and long-term Bitcoin price movements using machine learning and time-series forecasting techniques.
2. **Market Sentiment Analysis**: Analyze sentiment from news articles, social media platforms, and financial reports using NLP techniques.
3. **Risk Analysis**: Evaluate risks in Bitcoin trading through volatility metrics, drawdowns, and historical trends.
4. **Trading Signal Generation**: Generate buy, sell, and hold signals based on predictive models and market conditions.
5. **Interactive Dashboards**: Visualize Bitcoin price trends, trading signals, and risk levels using interactive dashboards.
6. **Automated Trading**: Implement an automated trading system to execute trades in real-time based on model insights.

## Data Collection and Preprocessing

### Data Sources
- **Price Data:**
  - Collected historical Bitcoin price data (open, high, low, close, volume) using the **Yahoo Finance API** via the `yfinance` library.
  - This data provides comprehensive information on market movements over time.

- **Circulating Supply:**
  - Retrieved the circulating supply of Bitcoin using the **CoinGecko API**.
  - The circulating supply is crucial for calculating the Market Capitalization, which is derived by multiplying the closing price with the circulating supply.

- **Sentiment Analysis Data:**
  - Sourced data for sentiment analysis using the **Reddit API**.
  - This data captures user opinions, discussions, and trends on Reddit, providing insights into market sentiment that can influence Bitcoin price movements.
 
### Data Cleaning
**Price Data Cleaning Steps**
- Convert 'Date' column to datetime with UTC awareness.
- Remove timezone information.
- Set 'Date' as the index.
- Sort the data chronologically.
- Remove unnecessary columns.

**Sentiment Analysis Data Cleaning Steps**
- Normalize text (convert to lowercase).
- Remove punctuation and special characters.
- Remove stopwords.
- Tokenize the text.

### Feature engineering

- **Date Features (Year, Month, Day, Quarter, Weekday):**
    - Capture seasonality and cyclical patterns inherent in time-series data.
    - Identify trends that vary by specific time periods (e.g., monthly or quarterly trends, weekend effects).

- **Daily Return:** 
    - Measures the percentage change in the closing price from one day to the next.
    - Provides insights into daily market volatility and short-term performance.

- **Exponential Moving Averages (EMA):**
  - **EMA_7:**   
      - Serves as a short-term trend indicator by smoothing price fluctuations over 7 days.
      - Highlights immediate market momentum.
  - **EMA_30:**   
      - Acts as a longer-term trend indicator by smoothing price data over 30 days.
      - Helps detect sustained trends and filters out short-term noise.

- **Relative Strength Index (RSI):** 
    - A momentum oscillator that measures the speed and change of price movements.
    - Identifies overbought (typically above 70) or oversold (typically below 30) conditions, aiding in predicting potential reversals.

- **Bollinger Bands:** 
    - Consist of a moving average (commonly a 20-day SMA) with upper and lower bands calculated using standard deviations.
    - Indicate volatility and help identify overbought or oversold conditions, as well as potential price breakouts.
 
### Data Splitting

- **Chronological Order:**  
  The dataset is first sorted by the `Date` column to maintain the natural time sequence.

- **80/20 Split:**  
  - **Training Set (80%):** Contains the earliest 80% of the data for learning historical trends.
  - **Test Set (20%):** Contains the most recent 20% of the data to evaluate future predictions.

This approach ensures the model is trained on past data and tested on future data, thereby preventing look-ahead bias.

### Feature Scaling

 **Standard Scaler:**  
     This involves scaling the data so that each feature has a mean of 0 and a standard deviation of 1. This step is crucial for algorithms sensitive to the scale of input data, ensuring that no single feature dominates due to its magnitude.

### Checking for Stationarity:
**Visual Inspection (Rolling Mean & Standard Deviation):**  
![image](https://github.com/user-attachments/assets/023abafb-c221-466c-b3b5-ded6b156e080)
The rolling mean and standard deviation fluctuate over time, the data is  non-stationary.

**Augmented Dickey-Fuller (ADF) Test:**  
Statistical test to check for a unit root in the series.
ADF Statistic (Original): 0.2040
p-value (Original): 0.9725
Interpretation: The original series is Non-Stationary.

**Making the Data Stationary:**
Making the data stationary is an important preprocessing phase of the examination of time 
series since it is beneficial to stabilize the statistical features of the data and enhance the 
forecasting model performance.

**Log Transformation:**  
The Bitcoin price data exhibits an exponential-like growth, with prolonged periods of relatively low values followed by sharp increases. This results in high variance, making trend analysis difficult.

To address this, I applied a log transformation (log(price)) for the following reasons:
- **Stabilizing Variance** 📉: The original data shows large fluctuations, especially in later years. Log transformation helps **normalize these variations**.  
- **Enhancing Trend Visibility** 📊: Without transformation, the earlier values appear almost flat compared to later spikes. Log transformation allows for a **clearer view of long-term trends**.  
- **Reducing the Impact of Outliers** ⚠️: Extreme price spikes dominate the scale in the original data. Applying a log transformation **compresses these values**, making patterns in the data more apparent.
![image](https://github.com/user-attachments/assets/104fc55a-eb2e-410c-81b8-57a15d62c3cf)

**Differencing:**  
Differencing the data removes trends and any remaining non-stationarity. This step involves subtracting the previous observation from the current observation, which helps in achieving a stationary series.
ADF Statistic (Differenced): -62.8500
p-value (Differenced): 0.0000
Interpretation: The differenced series is Stationary.

**Visualizing Differenced Data:**
![image](https://github.com/user-attachments/assets/02b0e924-3aef-4617-9add-daf070369a13)

**Seasonality Analysis:**
![image](https://github.com/user-attachments/assets/e7b93bc0-1a5a-4926-a8da-6adfcc496281)
**1️⃣ Original Time Series (Top Panel: "Close")**
- The Bitcoin price follows an exponential growth pattern with large fluctuations.  
- This confirms the need for log transformation or differencing to stabilize variance.  

**2️⃣ Trend Component**  
- There's a clear upward trend with some flattening and dips (e.g., around 2021–2022).  
- This suggests that differencing was needed to remove the trend for ARIMA modeling.  

**3️⃣ Seasonal Component**
- There is a strong seasonal pattern, repeating consistently every year.  
- This suggests that SARIMA (Seasonal ARIMA) might be a better choice than ARIMA.  

**4️⃣ Residual Component**
- The residuals appear to be somewhat stationary.  
- However, variance increases around 2021-2022, which might indicate volatility clustering.  

![image](https://github.com/user-attachments/assets/9c434f86-ca7f-460a-b03c-e6c42ca85f69)
**1️⃣ Transformed Time Series ("Log_Close_Diff")**  
- The original exponential trend has been **removed**, making fluctuations more stable.  
- Differencing has eliminated the strong upward trend, helping make the series **more stationary**.  
- However, some **volatility spikes remain**, indicating that further modeling may be required.  

**2️⃣ Trend Component**  
- The trend is now more **stable**, with gradual fluctuations instead of exponential growth.  
- The downward movement around **2021–2022** aligns with market corrections.  
- This suggests that **differencing was effective** but still retains some underlying structure.  

**3️⃣ Seasonal Component**  
- A **repeating pattern** is visible, meaning seasonality is still present even after transformation.  
- This suggests that **SARIMA (Seasonal ARIMA) may be preferable** over standard ARIMA.  
- The periodic nature (possibly **weekly, monthly, or yearly**) should be analyzed further using ACF/PACF plots.  

**4️⃣ Residual Component**  
- Residuals look **more stationary**, meaning most trends and seasonality have been removed.  

**ACF and PACF Plots:**  
PACF (Partial Autocorrelation Function) and ACF (Autocorrelation Function) plots are 
useful for analyzing and determining the order of moving average (MA) and autoregressive 
(AR) components in an ARIMA model. The ACF and PACF charts can be used to establish 
the right values for the MA and AR variables in an ARIMA model. The ACF plot aids in 
the identification of MA terms by noticing significant spikes at certain lags, whereas the 
PACF plot aids in the identification of AR terms by observing significant spikes at specific 
lags.
![image](https://github.com/user-attachments/assets/4cc54c84-a628-4c4f-8ae7-9a8584ed4ca1)
![image](https://github.com/user-attachments/assets/4afe1d65-81d6-4390-9597-00ffb7b235d9)
![image](https://github.com/user-attachments/assets/56d8112d-340d-4e25-9cff-d342b87442ca)

These preprocessing steps ensured that the data was well-prepared—scaled, stationary, and seasonally decomposed—so that the subsequent modeling could be both effective and reliable.
 
2. **Model Development**:
   - Train time-series forecasting models (e.g., ARIMA, LSTM) for price prediction.
   - Implement classification models to generate trading signals.
   - Combine sentiment analysis results with technical indicators to enhance prediction accuracy.

3. **Model Evaluation**:
   - Evaluate models using metrics like MAE, RMSE for price prediction and accuracy for trading signals.
   - Conduct cross-validation and hyperparameter tuning to optimize performance.

4. **Data Visualization and Dashboard Development**:
   - Create dashboards to display historical trends, predicted prices, trading signals, and risk assessments.
   - Include visualizations of volatility, drawdowns, and reward-to-risk ratios.

5. **Model Deployment**:
   - Deploy the system using Flask or Streamlit, enabling users to access predictions and trading signals via a web app.
   - Integrate real-time trading alerts using Twilio or email notifications.


## Technologies Used

- **Programming Languages**: Python
- **Machine Learning Libraries**: 
  - Scikit-learn (classification models, trading signals)
  - XGBoost, LightGBM (for predictive accuracy)
  - TensorFlow/Keras (LSTM, GRU, Transformer models for forecasting)
- **Data Collection & Preprocessing**:
  - APIs: Binance, CoinGecko, Alpha Vantage
  - Libraries: Pandas, NumPy, Requests
  - Time-Series: Statsmodels, pmdarima
- **NLP Tools**: spaCy, NLTK, Hugging Face Transformers
- **Data Visualization**: 
  - Tableau (dashboards)
  - Matplotlib, Seaborn, Plotly (custom visualizations)
- **Web Development**: Flask/Streamlit (for web application deployment)
- **Real-time Alerts**: Twilio (SMS/Email notifications)
