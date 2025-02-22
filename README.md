# Cryptocurrecncy Price Prediction

## 1. Introduction

This project aims to build a Cryptocurrency Price Prediction system to help traders and investors make data-driven decisions. Cryptocurrency markets are highly volatile, and predicting price movements can minimize risks, maximize profits, and enhance trading strategies.

## 2. Problem Statement 

Cryptocurrency prices are highly volatile, making it difficult for traders and investors to make informed decisions. This volatility leads to financial losses, market manipulation, and investment uncertainty. The project aims to address these challenges by predicting future prices using machine learning and statistical analysis, providing data-driven insights, and reducing uncertainty in cryptocurrency trading.

## 3. Objectives

This project aims to achieve the following goals:

- **Analyzing Historical Price Trends** – Study past price trends to understand market behavior and identify patterns.
- **Implementing Machine Learning and Time series Models for Prediction** – Develop and train models like LSTMs, Random Forest, and ARIMA to predict future cryptocurrency prices.
- **Evaluating Model Accuracy and Improving Predictions** – Assess the performance of different models and fine-tune them to improve prediction accuracy.

## 4. Data Collection

### Data Source

The data for this project is sourced using the following APIs:

- **Price Data:**  
  - **Source:** Yahoo Finance API via the `yfinance` library  
  - **Data:** Open, high, low, close, and volume metrics.

- **Circulating Supply:**
 - **Source:** CoinGecko API
 - **Data:** The circulating supply is used to calculate the Market Cap of each coin, by multiplying the Close price by the circulating supply during that time period.

### Features in the Dataset

The dataset includes the following key features:

- **Open** – The price at which the cryptocurrency opened during a specific time period.
- **Close** – The price at which the cryptocurrency closed during the specific time period.
- **High** – The highest price during the time period.
- **Low** – The lowest price during the time period.
- **Volume** – The total number of units traded during the time period.
- **Market Cap** - The total market value of the cryptocurrency during the time period.
- **Exponential Moving Averages (EMA):**
  - **EMA_50** – Short-term trend indicator (50-day EMA).
  - **EMA_200** – Long-term trend indicator (200-day EMA).
- **Daily Return:** Measures the percentage change in closing prices between consecutive days.
- **Relative Strength Index (RSI):** Identifies overbought (>70) or oversold (<30) conditions.

## 5. Exploratory Data Analysis

### Cryptocurrency Price Statistics 

| Coin | Count | Mean       | Std        | Min      | 25%       | 50%       | 75%       | Max       |
|------|-------|------------|------------|----------|-----------|-----------|-----------|-----------|
| BTC  | 3807  | 20588.04   | 23711.62   | 178.10   | 1278.03   | 9508.99   | 32941.57  | 106146.27 |
| ETH  | 2658  | 1520.86    | 1235.34    | 84.31    | 274.32    | 1455.79   | 2472.06   | 4812.09   |
| SOL  | 1775  | 72.48      | 71.12      | 0.52     | 18.25     | 35.56     | 136.75    | 261.87    |

**Summary:**

- The **mean closing price** for **BTC** is **$20,588**, with a **high standard deviation** of **$23,711**, indicating significant volatility.
- **BTC** prices range from a **minimum of $178** to a **maximum of $106,146**, showing a large price spread.
- The **median closing price** for **BTC** is **$9,508**, much lower than the **mean**, indicating a **right-skewed distribution**.
  
- For **ETH**, the **mean price** is **$1,520**, and its price range is from **$84** to **$4,812**, with a **standard deviation** of **$1,235**.
- The **median closing price** for **ETH** is **$1,455**, suggesting a mild **right-skewed distribution**.

- **SOL** has a **mean closing price** of **$72.48**, and a **standard deviation** of **$71.12**, indicating high volatility.
- The **price range for SOL** spans from **$0.52** to **$261.87**, showing considerable fluctuations, though smaller than **BTC** and **ETH**.
- The **median closing price** for **SOL** is **$35.56**, reinforcing a **right-skewed distribution**.

These observations highlight the volatility and price distributions for **BTC**, **ETH**, and **SOL**, with **BTC** showing the widest price range and highest volatility.
 
### Frequency Distribution

![image](https://github.com/user-attachments/assets/1a12dba9-b3a9-43bc-b7a4-85276e753dce)

- The close price shows a **right-skewed distribution**, indicating that lower prices were more frequent.
- There are **multiple peaks**, suggesting different historical price levels or market phases.  

### Correlation Martix

![image](https://github.com/user-attachments/assets/2c4751e0-4b73-4b84-a3ac-862efc82e0df)

## 6. Data Preprocessing

To prepare the dataset for analysis and model training, the following preprocessing steps were performed:

- **Feature Engineering:** I created additional features such as Exponential Moving Averages (EMA_50, EMA_200), Daily Return, and RSI to gain deeper insights into the price trends.
- **Feature Scalling:** Since the 'Close' prices was not normally distributed, I applied the **Min-Max Scaler** to normalize the data. This transformed the values into a range between 0 and 1, ensuring the model could better learn from the data.
- **Splitting Data into Training & Testing Sets:** The data was split into **80% training** and **20% testing**. This division allowed the model to learn from the majority of the data while being evaluated on a separate testing set, ensuring it can generalize well to new, unseen data.


## 7. Data Analysis

### Closing Price Over Time

![image](https://github.com/user-attachments/assets/93fb0b22-489e-40ab-bb68-aaca3e100d6f)

![image](https://github.com/user-attachments/assets/567309a6-695a-461b-ac30-ce357237217b)


**1️⃣ Bitcoin (BTC) Dominates Price Trends**
- **BTC (orange)** remains the highest-priced cryptocurrency.
- Surged past **$100,000 in 2024**, showing strong market confidence.
- **Historical peaks in 2017, 2021, and 2024** indicate repeated bull cycles.

**2️⃣ Ethereum (ETH) Shows Moderate Growth**
- **ETH (blue)** has a much lower price range compared to BTC.
- Peaked around **$5,000** in previous bull cycles.
- **Gradual upward trend**, showing solid adoption.

**3️⃣ Solana (SOL) Remains Relatively Lower in Price**
- **SOL (green)** shows price spikes after 2021 but remains below BTC & ETH.

### Market Capitalization

![image](https://github.com/user-attachments/assets/f8492bd6-fff0-4b25-9de9-a6b1bde9f2ae)


**1️⃣ Bitcoin (BTC) Leads the Market**
- **BTC (orange)** has the highest market capitalization, peaking above **$2 trillion**.
- Significant **growth during bull runs** (2017, 2021, 2024).
- **Recent 2024 surge** suggests renewed investor confidence.

**2️⃣ Ethereum (ETH) Shows Strong Growth**
- **ETH (blue)** follows BTC but at a lower magnitude.
- Peaked around **$500 billion** in 2021 but remains **steadily increasing**.
- Indicates **strong network utility and adoption**.

**3️⃣ Solana (SOL) Gains Traction**
- **SOL (green)** had a late start but saw significant growth post-2021.
- **Smaller market cap** compared to BTC & ETH but shows steady **adoption and resilience**.

**4️⃣ Market Cycles Are Clearly Visible**
- **Boom and bust cycles** are evident (2021 bull run, 2022 bear market).
- **Post-2023 recovery** shows renewed market interest.

### Trading Volume Analysis

![image](https://github.com/user-attachments/assets/571be646-4d9c-463d-87f6-07b6371a8ef0)


1️⃣ Bitcoin (BTC) Dominates Trading Volume  
- **BTC (orange)** has the highest trading volume over time, especially during market peaks.  
- Major **spikes align with market cycles** (e.g., 2021 bull run).  

2️⃣ Ethereum (ETH) Has Consistently High Volume  
- **ETH (blue)** follows BTC’s trend but at a lower scale.  
- Shows **sustained liquidity**, indicating strong investor interest.  

3️⃣ Solana (SOL) Gained Traction Post-2020  
- **SOL (green)** had minimal trading before 2020 but grew rapidly.  
- Lower volume than BTC & ETH, but **trading activity is increasing**.  

 4️⃣ Volume Spikes Correlate with Market Events  
- **2021:** Crypto bull run → **Highest trading activity ever recorded**.  
- **2022:** Market crash → **Sharp spikes, indicating panic selling**.  
- **2024:** Volume stabilizes but remains **volatile, especially for BTC**.

### Moving Averages (EMA)

![image](https://github.com/user-attachments/assets/5f68a070-e34c-4bde-86b3-1a9ac4580a82)

### Volatility Analysis Using Rolling Standard Deviation

Volatility measures how much prices fluctuate over time. Higher volatility indicates higher risk but also higher potential returns.

![image](https://github.com/user-attachments/assets/f5caeb7d-c419-4f67-aeea-e4df04ac0d12)


 1️⃣ Bitcoin (BTC) Shows the Most Stability  
- BTC (orange) maintains relatively low volatility over time.  
- This suggests it is more **established** and less reactive to short-term market movements.  

 2️⃣ Ethereum (ETH) and Solana (SOL) Are More Volatile  
- ETH (blue) experiences **moderate fluctuations**, especially during market shifts.  
- SOL (purple) has the **highest volatility**, with frequent sharp spikes.  
- Post-2021, **SOL's volatility exceeds ETH & BTC**, indicating **higher speculative activity**.  

3️⃣ Volatility Spikes Align with Major Market Events  
- 2018: Crypto market crash → Sudden surge in volatility.  
- 2020: Pandemic-driven uncertainty → Increased market swings.  
- 2021: **Bull run & corrections** → Highest volatility levels observed.

### Relative Strength Index

![image](https://github.com/user-attachments/assets/9dee60b9-23b5-455b-87ac-80c2d21959c4)


## 8. Model Development
My approach involves testing and comparing several types of models to determine the best fit for cryptocurrency price prediction:

- Machine Learning Models: Random Forest & XGBoost
  - Capture complex, non-linear relationships
- Time Series Models: ARIMA, SARIMA & Prophet
  - Model trends, seasonality, and cyclical patterns in the data.
- Deep Learning Models: Long Short Term Memory (LSTM)
  - Leverage recurrent neural networks to capture long-term dependencies in    sequential data.

### Machine Learning Models

**Random Forest**
- Random Forest algorithm works by creating a collection of multiple decision trees, each trained on a slightly different random subset of the data.
- It then combines their predictions to reach a final result, effectively averaging the outputs of these trees to produce a more accurate prediction than any single tree alone; this approach is called "ensemble learning" and helps to reduce overfitting by introducing diversity among the trees.

 ![image](https://github.com/user-attachments/assets/7578abd1-e7e2-4fbb-92d3-7c6bb9165d83)

 Hyperparameters
- n_estimators = 200 trees for better accuracy and reduced variance.
- max_depth = 10: Limits tree depth to prevent overfitting.
- max_features = 'sqrt': Randomly selects features to add diversity and reduce overfitting.
- random_state = 42: Ensures reproducibility and consistency.

**XGBoost (eXtreme Gradient Boosting)**
- XGBoost works by sequentially building a series of decision trees, where each new tree learns from the errors made by the previous trees, effectively correcting the residuals and improving the overall prediction accuracy.
- It is a supervised learning algorithm that uses gradient descent to optimize the model, allowing it to handle large datasets efficiently and achieve high performance in both classification and regression tasks.

![image](https://github.com/user-attachments/assets/7f83aa96-ceb1-4e19-ae42-5f259b9ac26f)

Hyperparameters
- n_estimators = 200 trees for improved accuracy.
- learning_rate = 0.05 to prevent overfitting.
- max_depth = 7 to control tree complexity.
- L2 (0.5) ridge regularization to prevent any single feature from dominating the model, since the dataset has highly correlated features.
- Random state of 42 for consistent results.

#### Machine learning Model Evaluation
<div style="display: flex; justify-content: center;">
  <img src="https://github.com/user-attachments/assets/0af46f87-e27c-4baa-b44a-d79e1cee3841" width="50%" style="margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/e8f5ea9a-fd5f-44e5-9754-979c4f3c9f59" width="50%">
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/40af424a-7389-4a6e-b47f-c9ebc3d735ee" width="30%">
  <img src="https://github.com/user-attachments/assets/4e5a8b74-43bb-4533-9606-ddbb26f9f4d0" width="30%">
</div>

#### Model Predictions
<div style="display: flex; justify-content: center;">
  <img src="https://github.com/user-attachments/assets/5b9ef78b-b361-4043-94e6-e9d1441ddb13" width="45%" style="margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/5055fd2e-f355-4e03-94ad-f0ab7982f206" width="45%">
</div>


### Time Series Models
#### Autoregressive Integrated Moving Average (ARIMA)
A statistical analysis model that predicts future values based on historical data. It has three main components: 
- AR (AutoRegressive): Uses past values to predict the future.
- I (Integrated): The differencing of raw observations to make the time series stationary.
- MA (Moving Average): Uses past errors to predict the future.
  
Key Parameters of ARIMA
- p: Number of past values (lags).
- d: Degree of differencing (to make data stationary).
- q: Moving average window size.
  
#### SARIMA (Seasonal AutoRegressive Integrated Moving Average)
An extension of ARIMA that accounts for seasonality in time series data. It is represented as:
SARIMA(p,d,q)×(P,D,Q,s)

![image](https://github.com/user-attachments/assets/25f53d72-2a9b-4ab1-adae-0da933d0c08d)

**Checking for Stationarity**

- **Visual Inspection (Rolling Mean & Standard Deviation):**  
  ![Rolling Statistics](https://github.com/user-attachments/assets/023abafb-c221-466c-b3b5-ded6b156e080)
  
- **ADF Test Results:**  
  - **Original Series:**  
    - ADF Statistic: 0.2040  
    - p-value: 0.9725  
    → *Non-stationary*
    
**Transformations for Stationarity**

- **Log Transformation:**
 - Applied log transformation to stabilize the variance and reduce the effect of large fluctuations in the data.
 - This helped smooth out exponential growth trends.

- **Differencing:**  
  - Applied first-order differencing to remove trends in the data.
  - Dickey-Fuller (ADF) Test to confirm that the transformed data met the stationarity assumption.

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/95046dac-709b-421d-bd69-ff08dac9b9d7" width="45%">
  <img src="https://github.com/user-attachments/assets/d2747dd4-f524-4a60-aaf8-6416b2a8ca0d" width="45%">
</div>

- **Differenced Series:**  
  - ADF Statistic: -62.8500  
  - p-value: 0.0000  
  → *Stationary*

### Autocorrelations (ACF) and Partial Autocorrelations (PACF) Plot
![image](https://github.com/user-attachments/assets/37769808-1563-4172-a7e0-dd2d91981372)

![image](https://github.com/user-attachments/assets/a7deada9-4cda-48a6-b6f1-caa1d6788761)

- ACF shows a sharp drop after lag 1 suggesting (Moving Average order = 1).
  
![image](https://github.com/user-attachments/assets/8f1c534f-5d3e-4480-82f6-68cc8e45eab9)

- PACF cuts off after lag 1 suggesting  (Autoregressive order = 1).
-  The ACF and PACF plot drops off quickly (with no strong pattern or slow decay), meaning the data is stationary after differencing.
-  This confirms that the differencing step (d = 1) was effective in removing trends.

### **Seasonality Analysis:**  

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/501b8ce2-6f36-487a-9114-0955f8219bef" width="45%">
  <img src="https://github.com/user-attachments/assets/dca441b1-4044-485f-b68e-8ee47b71feb0" width="45%">
</div>

### **Time Series Windowing (Sequence Generation)**

For LSTM models, we use a sliding window approach (look_back = 5) to create 3D tensors:
- **Samples:** Number of sequences.
- **Timesteps:** Length of the sliding window.
- **Features:** Number of features per timestep.

---

### **LSTM Model Architecture**

- **LSTM Layers:**  
  - **Layer 1:** 50 units, returns sequences, uses tanh activation with L2 regularization.
  - **Layer 2:** 40 units, final output state, tanh activation with L2 regularization.
  
- **Dropout Layers:**  
  0.3 dropout after each LSTM layer to mitigate overfitting.
  
- **Dense Layers:**  
  - 30-unit dense layer with ReLU activation.
  - Final dense layer with 1 unit for regression.

- **Training Details:**  
  - **Optimizer:** Adam with a learning rate of 0.0005  
  - **Loss Function:** Mean Squared Error (MSE)  
  - **Epochs:** Up to 100 with early stopping (patience = 10, restore best weights)  

  ![image](https://github.com/user-attachments/assets/fcde4ce9-27a3-4c25-84fe-fe95598f7249)

---

### **Data Splitting**

- **Chronological Order:**  
  Data is sorted by date to preserve temporal relationships.

- **80/20 Split:**  
  - **Training Set (80%):** Earliest data for model learning.  
  - **Test Set (20%):** Most recent data for evaluation.  
  ![Data Splitting](https://github.com/user-attachments/assets/4890909f-c3c7-41bf-b97c-669e3033f895)

---

## 9. Model Predictions

### **Random Forest**
- **Actual vs Predicted Prices:**
  ![image](https://github.com/user-attachments/assets/19c0219e-ea12-4dd2-b137-9eacc25564a6)

### **XGBoost**
- **Actual vs Predicted Prices:**  
  ![image](https://github.com/user-attachments/assets/718be5c9-da6c-4366-950b-443795705611)

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
  ![image](https://github.com/user-attachments/assets/bd371742-e43c-4c9c-8b9c-80892b4f343b)

### **LSTM**
- **Results:**  
  ![image](https://github.com/user-attachments/assets/0e9d6d2c-d174-47de-98be-0903172e3d12)
  
  ![image](https://github.com/user-attachments/assets/517cb312-e6ac-48a1-b753-257d219168ba)

  ![image](https://github.com/user-attachments/assets/6d238d5c-a5e3-47d5-ac7d-b9c677610d17)

---

## 10. Model Evaluation

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
| **LSTM**           | 2147.02    | 2832.39    |
| **Random Forest**  | 4761.72    | 11494.35   |
| **XGBoost**        | 5357.49    | 12270.71   |
| **Prophet**        | 11342.16   | 15621.58   |
| **ARIMA**          | 26449.34   | 34524.33   |
| **SARIMA**         | 15283.65   | 20759.14   |
| **Tuned SARIMA**   | 30186.72   | 38668.71   |

**LSTM Crypto model evaluation**

| **Model**          | **MAE**    | **RMSE**   |
|--------------------|------------|------------|
| **Bitcoin**        | 2147.02    | 2832.39    |
| **Ethereum**       |  125.98    | 174.23     |
| **Solana**         |  10.47     | 13.71      |

**Conclusion:**  
Based on these metrics, **LSTM** emerges as the best-performing model for forecasting Bitcoin prices with the lowest MAE and RMSE values.

---

## 11. Model Deployment: 

Deployed the model using streamlit 

  
