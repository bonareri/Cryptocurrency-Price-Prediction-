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

###  Methodologies Used for Analysis
The project focuses on a regression task—predicting Bitcoin prices—using a variety of models.

**Random Forest:** 
An ensemble learning method that effectively captures complex, non-linear relationships. It's particularly useful for regression tasks where the goal is to predict continuous outcomes, such as stock or cryptocurrency prices.

 **ARIMA (AutoRegressive Integrated Moving Average):**
 An established time series analysis model that predicts future values based on historical data.
 It is composed of 3 main parts: 
 - AR (AutoRegressive): Uses past values to predict the future.
 - I (Integrated): The differencing of raw observations to make the time series stationary.
 - MA (Moving Average): Uses past errors to predict the future. 
 
 **SARIMA (Seasonal ARIMA):** 
 An extension of ARIMA that incorporates seasonal patterns, making it ideal for data with regular periodic fluctuations.

 **PROPHET**
Prophet is an open-source forecasting tool developed by Facebook, designed to deliver high-quality forecasts for time series data with complex seasonal patterns and multiple trend changes.

**LSTM (Extended Short-term Memory)**
LSTM is a type of recurrent neural network (RNN) designed to overcome the vanishing gradient problem. It uses internal gates to regulate information flow, enabling effective backpropagation through time (BPTT).

An LSTM network is built using LSTM cells, each containing the following components:

- **Input Gate:**  
  Controls how much new information is stored in the memory cell. It uses a sigmoid activation on the current input and previous hidden state, outputting values between 0 and 1.

- **Forget Gate:**  
  Decides which information to discard from the memory cell. It processes the current input and previous hidden state using a sigmoid activation function.

- **Memory Cell:**  
  Stores information over time by integrating signals from both the input and forget gates. It uses a tanh activation function to generate a new candidate cell state.

- **Output Gate:**  
  Determines the output of the LSTM cell. It applies a sigmoid function to the current input and previous hidden state to decide which information to pass on.

https://th.bing.com/th?id=OIP.1ylt72UVW-wTRr382T17TgHaFI&w=300&h=208&c=8&rs=1&qlt=90&o=6&dpr=1.5&pid=3.1&rm=2

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
 
### Feature Scaling

 **Standard Scaler:**  
     This involves scaling the data so that each feature has a mean of 0 and a standard deviation of 1. This step is crucial for algorithms sensitive to the scale of input data, ensuring that no single feature dominates due to its magnitude.

### Implementation 
#### Dataset 
The Bitcoin dataset comprises 3779 rows with the following key columns (with `Date` as the index):
- **Open:**  
  The price at which Bitcoin started trading on that particular day.
- **High:**  
  The highest price reached by Bitcoin during the trading day.
- **Low:**  
  The lowest price at which Bitcoin traded during the day.
- **Close:**  
  The price at which Bitcoin ended trading for the day.
- **Volume:**  
  The total number of Bitcoin traded during the day, providing insight into market activity and liquidity.
- **Market Cap:**  
  The market capitalization for Bitcoin on that day, calculated as the closing price multiplied by the circulating supply.

#### Visualizing the data
![image](https://github.com/user-attachments/assets/d54fda0d-fb59-4ebf-8c09-13fc1e7b62e0) 

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

**Time Series Windowing (Sequence Generation)** 
LSTM models learn from sequential patterns instead of treating each data point independently. 
Time series windowing transforms raw time series data into input-output pairs so that the model can learn from historical data and make accurate future predictions.
**Sliding Window Approach**
I set a look-back period (look_back = 5), meaning the model will use the past 5 days' prices to predict the next day's price.This helps capture short-term trends in the data.

**Reshaping for LSTM:**
   - Reshape the sequenced data into a 3D tensor with dimensions:  
     - **Samples:** Number of sequences generated.
     - **Timesteps:** The width of the sliding window (e.g., 5 in this case).
     - **Features:** The number of features per timestep.

**LSTM Model Architecture:**
- **LSTM Layers:**
    - **First LSTM Layer:**  
      - 40 units, returns sequences for subsequent LSTM layers.
      - Uses tanh activation and includes L2 regularization (`l2(0.001)`) to penalize large weights.
    - **Second LSTM Layer:**  
      - 30 units, does not return sequences (outputs final state).
      - Also uses tanh activation and L2 regularization.
  - **Dropout Layers:**  
    - Dropout rate of 0.3 after each LSTM layer to mitigate overfitting by randomly deactivating neurons during training.
  - **Dense Layers:**
    - A Dense layer with 20 units and ReLU activation to capture non-linear patterns.
    - Final Dense layer with 1 unit for regression (predicts a continuous value).
    
- **Compilation and Training:**  
  - **Optimizer:**  
    - Adam optimizer with a reduced learning rate (`0.0005`) for stable convergence.
  - **Loss Function:**  
    - Uses Mean Squared Error (MSE) as the loss function, which is standard for regression tasks.
  - **Training Configuration:**  
    - Training occurs over up to 100 epochs with a batch size of 64, balancing training speed and convergence.
    - Validation data is provided to monitor performance and trigger early stopping if necessary.

- **Early Stopping:**  
  - Prevents overfitting by halting training when the validation loss stops improving.
  - Training stopped at epoch 43 because the early stopping criteria were met.
  - **Configuration:**  
    - Monitors `val_loss`.
    - `patience=10`: Allows 10 epochs for improvement.
    - `restore_best_weights=True`: Ensures the model retains the best weights encountered during training.
      
### Data Splitting

- **Chronological Order:**  
  The dataset is first sorted by the `Date` column to maintain the natural time sequence.

- **80/20 Split:**  
  - **Training Set (80%):** Contains the earliest 80% of the data for learning historical trends.
  - **Test Set (20%):** Contains the most recent 20% of the data to evaluate future predictions.
![image](https://github.com/user-attachments/assets/4890909f-c3c7-41bf-b97c-669e3033f895)

### Model Predictions
**Random Forest**
![image](https://github.com/user-attachments/assets/2feca498-794d-489b-bdd8-09942ae42928)

**XGBoost**
![image](https://github.com/user-attachments/assets/d538bb94-4c6a-4718-b734-1607ddb23719)


ARIMA Results                                
==============================================================================
Dep. Variable:              Log_Close   No. Observations:                 3023
Model:                 ARIMA(1, 0, 1)   Log Likelihood                5546.001
Date:                Sat, 08 Feb 2025   AIC                         -11084.002
Time:                        20:27:16   BIC                         -11059.946
Sample:                    10-06-2014   HQIC                        -11075.353
                         - 01-14-2023                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          8.3953      0.856      9.807      0.000       6.717      10.073
ar.L1          0.9995      0.000   2863.532      0.000       0.999       1.000
ma.L1         -0.0191      0.011     -1.670      0.095      -0.042       0.003
sigma2         0.0015   1.53e-05     97.511      0.000       0.001       0.002
===================================================================================
Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):             15680.65
Prob(Q):                              0.94   Prob(JB):                         0.00
Heteroskedasticity (H):               1.17   Skew:                            -0.79
Prob(H) (two-sided):                  0.01   Kurtosis:                        14.04
===================================================================================

**Summary**

- The **AR(1) coefficient is highly significant**, indicating strong autocorrelation. However, the **MA(1) coefficient is not statistically significant**.  
- The **residuals are not normally distributed**, but the model does not suffer from **autocorrelation or heteroskedasticity**.  
- The **AIC is very low (-10703.468)**, suggesting a **good model fit**.

**Actual vs Predicted Prices**
![image](https://github.com/user-attachments/assets/538a5c17-332f-4bf5-8eb5-6fba4045cca9)

**SARIMA**
SARIMAX Results                                      
==========================================================================================
Dep. Variable:                          Log_Close   No. Observations:                 3023
Model:             SARIMAX(1, 0, 1)x(1, 0, 1, 12)   Log Likelihood                5522.789
Date:                            Sat, 08 Feb 2025   AIC                         -11035.578
Time:                                    20:27:50   BIC                         -11005.531
Sample:                                10-06-2014   HQIC                        -11024.772
                                     - 01-14-2023                                         
Covariance Type:                              opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          1.0001   7.96e-05   1.26e+04      0.000       1.000       1.000
ma.L1         -0.0180      0.012     -1.553      0.120      -0.041       0.005
ar.S.L12      -0.1102      0.735     -0.150      0.881      -1.552       1.331
ma.S.L12       0.0982      0.736      0.133      0.894      -1.345       1.542
sigma2         0.0015   1.55e-05     96.022      0.000       0.001       0.002
===================================================================================
Ljung-Box (L1) (Q):                   0.03   Jarque-Bera (JB):             15636.00
Prob(Q):                              0.87   Prob(JB):                         0.00
Heteroskedasticity (H):               1.14   Skew:                            -0.80
Prob(H) (two-sided):                  0.04   Kurtosis:                        14.05
===================================================================================
![image](https://github.com/user-attachments/assets/54e9f8a0-04fa-45f6-a9f2-c7d5ce7e519e)

**Hyperparameter Tuning**
SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                 3023
Model:               SARIMAX(0, 1, 0)   Log Likelihood                5549.939
Date:                Sat, 08 Feb 2025   AIC                         -11095.879
Time:                        20:28:56   BIC                         -11083.851
Sample:                    10-06-2014   HQIC                        -11091.554
                         - 01-14-2023                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept      0.0014      0.001      1.911      0.056   -3.49e-05       0.003
sigma2         0.0015   1.54e-05     96.740      0.000       0.001       0.002
===================================================================================
Ljung-Box (L1) (Q):                   1.26   Jarque-Bera (JB):             15561.56
Prob(Q):                              0.26   Prob(JB):                         0.00
Heteroskedasticity (H):               1.18   Skew:                            -0.79
Prob(H) (two-sided):                  0.01   Kurtosis:                        14.01
===================================================================================
![image](https://github.com/user-attachments/assets/0d8e1b5d-8e8e-4520-b97d-5f318e961b1f)

**PROPHET**
![image](https://github.com/user-attachments/assets/ef468343-90ff-4a5b-bf14-cff7ab04d3db)

**LSTM Results**
![image](https://github.com/user-attachments/assets/dbb79603-2738-4cc7-9403-e5436bec0268)

![image](https://github.com/user-attachments/assets/6165d25e-d274-47dc-bca7-15c2be89effa)

## Model Evaluation
**Mean Absolute Error (MAE):**  
  - This metric represents the average absolute difference between predicted and actual values.
  - MAE is less sensitive to outliers compared to RMSE, providing a straightforward measure of prediction accuracy.

- **Root Mean Squared Error (RMSE):**  
  - **Penalty on Larger Errors:** RMSE gives higher weight to larger errors due to the squaring of differences, making it particularly useful when large deviations are undesirable.
  - **Standard Interpretation:** RMSE is in the same units as the target variable, making it directly comparable to the actual values.
 
### Best Model Selection
After evaluating multiple models, the performance metrics are summarized as follows:

- **Prophet:**
  - MAE: 1271.30
  - RMSE: 1722.81

- **LSTM:**
  - MAE: 2108.59
  - RMSE: 2777.53

- **Random Forest:**
  - MAE: 4280.44
  - RMSE: 11070.36

- **XGBoost:**
  - MAE: 4794.99
  - RMSE: 11542.46

- **ARIMA:**
  - MAE: 26007.51
  - RMSE: 33977.92

- **SARIMA:**
  - MAE: 14011.45
  - RMSE: 19170.45

- **Tuned SARIMA:**
  - MAE: 29707.52
  - RMSE: 38076.21

**Conclusion:** 
Based on the evaluation metrics provided, **Prophet** is the best-performing model for this task with a lower MAE and RMSE values indicating that it is the most accurate and reliable model for forecasting in this project.

4. **Data Visualization and Dashboard Development**:
   - Create dashboards to display historical trends, predicted prices, trading signals, and risk assessments.
   - Include visualizations of volatility, drawdowns, and reward-to-risk ratios.

5. **Model Deployment**:
   - Deploy the system using Flask or Streamlit, enabling users to access predictions and trading signals via a web app.
   - Integrate real-time trading alerts using Twilio or email notifications.
