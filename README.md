# Bitcoin Price Prediction and Risk Analysis

---

## 1. Introduction

Bitcoin is a digital currency operating on a decentralized blockchain system, independent of banks or governments. Despite its growing adoption as an investment and payment method, Bitcoin is notoriously volatile—its price can change rapidly based on news, market trends, and investor sentiment.

For traders and investors, this volatility offers both opportunities and risks. Without robust analytical tools, predicting Bitcoin’s price movements can be challenging, potentially leading to poor trading decisions and financial losses. Traditional trading methods, which often rely on experience and intuition, may not suffice in such a fast-paced market.

---

## 2. Project Overview

This project leverages multiple time-series forecasting models and sentiment analysis techniques to:

- **Forecast Bitcoin Prices**
- **Assess Market Risks**
- **Generate Actionable Trading Signals**

By integrating classical forecasting methods with modern deep learning and NLP techniques, the system provides data-driven insights for more informed trading decisions.

---

## 3. Project Objectives

1. **Bitcoin Price Prediction:**  
   Predict short-term and long-term price movements using machine learning and time-series forecasting techniques.

2. **Market Sentiment Analysis:**  
   Analyze sentiment from news articles, social media, and financial reports using NLP techniques.

3. **Risk Analysis:**  
   Evaluate trading risks through volatility metrics, drawdowns, and historical trends.

4. **Trading Signal Generation:**  
   Generate buy, sell, and hold signals based on predictive models and market conditions.

---

## 4. Methodologies Used for Analysis

This project centers on a regression task—predicting Bitcoin prices—using a variety of models:

### **Random Forest**
- **Description:**  
  An ensemble method that captures complex, non-linear relationships.
- **Application:**  
  Effective for predicting continuous outcomes like cryptocurrency prices.

### **ARIMA (AutoRegressive Integrated Moving Average)**
- **Components:**  
  - **AR (AutoRegressive):** Uses past values to predict the future.
  - **I (Integrated):** Differencing to achieve stationarity.
  - **MA (Moving Average):** Uses past errors for prediction.

### **SARIMA (Seasonal ARIMA)**
- **Extension:**  
  Incorporates seasonal patterns to handle periodic fluctuations.

### **PROPHET**
- **Developed by:**  
  Prophet is a prominent open-source library for time series forecasting created by Facebook. 
- **Strength:**  
  Handles complex seasonal patterns and multiple trend changes.

### **LSTM (Long Short-Term Memory)**
- **Type:**  
  A recurrent neural network (RNN) that overcomes the vanishing gradient problem.
- **Key Components:**  
  - **Input Gate:** Regulates new information storage.
  - **Forget Gate:** Discards unnecessary information.
  - **Memory Cell:** Retains long-term data.
  - **Output Gate:** Determines the final output.

> **Visual Reference:**  
> [LSTM Gate Diagram](https://th.bing.com/th?id=OIP.1ylt72UVW-wTRr382T17TgHaFI&w=300&h=208&c=8&rs=1&qlt=90&o=6&dpr=1.5&pid=3.1&rm=2)

---

## 5. Data Collection and Preprocessing

### **Data Sources**

- **Price Data:**  
  - **Source:** Yahoo Finance API via the `yfinance` library  
  - **Data:** Open, high, low, close, and volume metrics.

- **Circulating Supply:**  
  - **Source:** CoinGecko API  
  - **Usage:** To calculate Market Capitalization (closing price × circulating supply).

- **Sentiment Analysis Data:**  
  - **Source:** Reddit API  
  - **Usage:** Captures user opinions and trends impacting Bitcoin price.

---

### **Data Cleaning**

#### **Price Data Cleaning**
- Convert the 'Date' column to a datetime format with UTC awareness.
- Remove timezone information.
- Set 'Date' as the index.
- Sort data chronologically.
- Remove unnecessary columns.

#### **Sentiment Analysis Cleaning**
- Normalize text (convert to lowercase).
- Remove punctuation and special characters.
- Remove stopwords.
- Tokenize the text.

---

### **Feature Engineering**

- **Date Features:**  
  Extract year, month, day, quarter, and weekday to capture seasonality and cyclic patterns.

- **Daily Return:**  
  Measures the percentage change in closing prices between consecutive days.

- **Exponential Moving Averages (EMA):**  
  - **EMA_7:** Highlights short-term trends.  
  - **EMA_30:** Indicates longer-term trends.

- **Relative Strength Index (RSI):**  
  Identifies overbought (>70) or oversold (<30) conditions.

- **Bollinger Bands:**  
  - Uses a 20-day SMA with upper and lower bands based on standard deviations.
  - Indicates volatility and potential breakouts.

---

### **Feature Scaling**

- **Standard Scaler:**  
  Scales data to a mean of 0 and a standard deviation of 1—important for algorithms sensitive to feature magnitude.

---

## 6. Implementation

### **Dataset Overview**

The Bitcoin dataset contains **3779 rows** with key columns:
- **Open, High, Low, Close, Volume, Market Cap**

### **Data Analysis Visualizations**

- **Trends Over Time:**  
  ![Long-Term Uptrend](https://github.com/user-attachments/assets/721f1403-1d6f-4fa1-b3f7-5223467843fa)  
  *Bitcoin has experienced significant surges in 2017, 2021, and 2024.*

- **Yearly Price Trends:**  
  ![Bitcoin Price Trends by Year](https://github.com/user-attachments/assets/c743dc66-ec1c-450a-afb1-22cb978ab99c)

- **Monthly Trends:**  
  ![Monthly Trends](https://github.com/user-attachments/assets/68de251f-0dd3-4114-a3b1-cb51b844dda8)

- **Weekday Trends:**  
  ![Weekday Trends](https://github.com/user-attachments/assets/cf4a6890-3b5f-4572-bfd2-6a9cab59c48c)

- **Daily Returns Distribution:**  
  ![Daily Returns Distribution](https://github.com/user-attachments/assets/7e4fe556-b557-4bdc-9721-25d6b20775c4)

- **Moving Averages:**  
  ![Moving Averages](https://github.com/user-attachments/assets/0d498884-dbb9-4780-bfba-e41263c854c6)

- **RSI Trends:**  
  ![RSI](https://github.com/user-attachments/assets/3dc28530-b4ac-4d1a-a7b4-997ce74cafcb)

- **Bollinger Bands:**  
  ![Bollinger Bands](https://github.com/user-attachments/assets/d99b86fd-4c75-4a87-8b15-efa0e48302ba)

---

### **Checking for Stationarity**

- **Visual Inspection (Rolling Mean & Standard Deviation):**  
  ![Rolling Statistics](https://github.com/user-attachments/assets/023abafb-c221-466c-b3b5-ded6b156e080)
  
- **ADF Test Results:**  
  - **Original Series:**  
    - ADF Statistic: 0.2040  
    - p-value: 0.9725  
    → *Non-stationary*
    
  - **Differenced Series:**  
    - ADF Statistic: -62.8500  
    - p-value: 0.0000  
    → *Stationary*

---

### **Transformations for Stationarity**

- **Log Transformation:**  
  Stabilizes variance and reduces the impact of outliers.  
  ![Log Transformation](https://github.com/user-attachments/assets/104fc55a-eb2e-410c-81b8-57a15d62c3cf)

- **Differencing:**  
  Removes trends, further stabilizing the series.

- **Seasonality Analysis:**  
  ![Seasonal Decomposition](https://github.com/user-attachments/assets/e7b93bc0-1a5a-4926-a8da-6adfcc496281)

---

### **Time Series Windowing (Sequence Generation)**

For LSTM models, we use a sliding window approach (look_back = 5) to create 3D tensors:
- **Samples:** Number of sequences.
- **Timesteps:** Length of the sliding window.
- **Features:** Number of features per timestep.

---

### **LSTM Model Architecture**

- **LSTM Layers:**  
  - **Layer 1:** 40 units, returns sequences, uses tanh activation with L2 regularization.
  - **Layer 2:** 30 units, final output state, tanh activation with L2 regularization.
  
- **Dropout Layers:**  
  0.3 dropout after each LSTM layer to mitigate overfitting.
  
- **Dense Layers:**  
  - 20-unit dense layer with ReLU activation.
  - Final dense layer with 1 unit for regression.

- **Training Details:**  
  - **Optimizer:** Adam with a learning rate of 0.0005  
  - **Loss Function:** Mean Squared Error (MSE)  
  - **Epochs:** Up to 100 with early stopping (patience = 10, restore best weights)  
  ![Training & Validation Loss](https://github.com/user-attachments/assets/6165d25e-d274-47dc-bca7-15c2be89effa)

---

### **Data Splitting**

- **Chronological Order:**  
  Data is sorted by date to preserve temporal relationships.

- **80/20 Split:**  
  - **Training Set (80%):** Earliest data for model learning.  
  - **Test Set (20%):** Most recent data for evaluation.  
  ![Data Splitting](https://github.com/user-attachments/assets/4890909f-c3c7-41bf-b97c-669e3033f895)

---

## 7. Model Predictions

### **Random Forest**
- **Actual vs Predicted Prices:**  
  ![Random Forest Results](https://github.com/user-attachments/assets/2feca498-794d-489b-bdd8-09942ae42928)

### **XGBoost**
- **Actual vs Predicted Prices:**  
  ![XGBoost Results](https://github.com/user-attachments/assets/d538bb94-4c6a-4718-b734-1607ddb23719)

### **ARIMA**
- **Actual vs Predicted Prices:**  
  ![ARIMA Results](https://github.com/user-attachments/assets/538a5c17-332f-4bf5-8eb5-6fba4045cca9)

### **SARIMAX**
- **Actual vs Predicted Prices:**  
  ![SARIMAX Results](https://github.com/user-attachments/assets/54e9f8a0-04fa-45f6-a9f2-c7d5ce7e519e)

### **Hyperparameter Tuning (SARIMAX)**
- **Results:**  
  ![Tuned SARIMAX Results](https://github.com/user-attachments/assets/0d8e1b5d-8e8e-4520-b97d-5f318e961b1f)

### **PROPHET**
- **Results:**  
  ![Prophet Results](https://github.com/user-attachments/assets/ef468343-90ff-4a5b-bf14-cff7ab04d3db)

### **LSTM**
- **Results:**  
  ![LSTM Results](https://github.com/user-attachments/assets/dbb79603-2738-4cc7-9403-e5436bec0268)

---

## 8. Model Evaluation

### **Metrics**

- **Mean Absolute Error (MAE):**  
  - Represents the average absolute difference between predicted and actual values.
  - Less sensitive to outliers.

- **Root Mean Squared Error (RMSE):**  
  - Gives higher weight to larger errors.
  - In the same units as the target variable, allowing direct comparison.

### **Performance Summary**

| **Model**          | **MAE**    | **RMSE**   |
|--------------------|------------|------------|
| **Prophet**        | 1271.30    | 1722.81    |
| **LSTM**           | 2108.59    | 2777.53    |
| **Random Forest**  | 4280.44    | 11070.36   |
| **XGBoost**        | 4794.99    | 11542.46   |
| **ARIMA**          | 26007.51   | 33977.92   |
| **SARIMA**         | 14011.45   | 19170.45   |
| **Tuned SARIMA**   | 29707.52   | 38076.21   |

**Conclusion:**  
Based on these metrics, **Prophet** emerges as the best-performing model for forecasting Bitcoin prices with the lowest MAE and RMSE values.

---

## 9. Next Steps

1. **Dashboard Development:**  
   - Create interactive dashboards to display historical trends, predicted prices, trading signals, and risk assessments.
   - Include visualizations for volatility, drawdowns, and reward-to-risk ratios.

2. **Model Deployment:**  
   - Deploy the system using frameworks like Flask or Streamlit.
   - Integrate real-time trading alerts via Twilio or email notifications.
