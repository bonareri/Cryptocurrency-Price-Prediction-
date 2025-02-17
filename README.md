# Cryptocurrecncy Price Prediction
---

## 1. Introduction

This project aims to build a Cryptocurrency Price Prediction system to help traders and investors make data-driven decisions. Cryptocurrency markets are highly volatile, and predicting price movements can minimize risks, maximize profits, and enhance trading strategies.

Unlike traditional markets, cryptocurrencies operate 24/7 and are influenced by factors such as social media trends and global regulations. The project utilizes techniques such as:

- **Machine Learning Models** (LSTMs, Random Forest, XG-Boost, ARIMA, SARIMA, PROPHET)
- **Statistical Analysis** (Moving averages, time series analysis)

These techniques will provide accurate insights to aid cryptocurrency traders and investors.

---

## 2. Problem Statement 

Cryptocurrency prices are highly volatile, making it difficult for traders and investors to make informed decisions. This volatility leads to financial losses, market manipulation, and investment uncertainty. The project aims to address these challenges by predicting future prices using machine learning and statistical analysis, providing data-driven insights, and reducing uncertainty in cryptocurrency trading.

## 3. Objectives

This project aims to achieve the following goals:

- **Analyzing Historical Price Trends** – Study past price trends to understand market behavior and identify patterns.
- **Implementing Machine Learning Models for Prediction** – Develop and train models like LSTMs, Random Forest, and ARIMA to predict future cryptocurrency prices.
- **Evaluating Model Accuracy and Improving Predictions** – Assess the performance of different models and fine-tune them to improve prediction accuracy.

By accomplishing these objectives, the project will provide reliable and data-driven predictions for cryptocurrency price movements.


## 4. Data Collection and Preprocessing

### Data Source

The data for this project is sourced using the following APIs:

- **Price Data:**  
  - **Source:** Yahoo Finance API via the `yfinance` library  
  - **Data:** Open, high, low, close, and volume metrics.

### Features in the Dataset

The dataset includes the following key features:

- **Open** – The price at which the cryptocurrency opened during a specific time period.
- **Close** – The price at which the cryptocurrency closed during the specific time period.
- **High** – The highest price during the time period.
- **Low** – The lowest price during the time period.
- **Volume** – The total number of units traded during the time period.
- **Exponential Moving Averages (EMA):**
  - **EMA_50** – Short-term trend indicator (50-day EMA).
  - **EMA_200** – Long-term trend indicator (200-day EMA).
- **Daily Return:** Measures the percentage change in closing prices between consecutive days.
- **Relative Strength Index (RSI):** Identifies overbought (>70) or oversold (<30) conditions.

### Preprocessing Steps

To prepare the dataset for analysis and model training, the following preprocessing steps were performed:

- **Feature Engineering:** I created additional features such as Exponential Moving Averages (EMA_50, EMA_200), Daily Return, and RSI to gain deeper insights into the price trends.
- **Normalization/Standardization:** Since the 'Close' prices were not normally distributed, I applied the **Min-Max Scaler** to normalize the data. This transformed the values into a range between 0 and 1, ensuring the model could better learn from the data.
- **Splitting Data into Training & Testing Sets:** The data was split into **80% training** and **20% testing**. This division allowed the model to learn from the majority of the data while being evaluated on a separate testing set, ensuring it can generalize well to new, unseen data.

---

## 5. Exploratory Data Analysis

### Cryptocurrency Price Statistics 
**Bitcoin Data**

| Statistic | Close       | High        | Low         | Open        | Volume        |
|-----------|------------|------------|------------|------------|--------------|
| Count     | 3807.000000 | 3807.000000 | 3807.000000 | 3807.000000 | 3.807000e+03 |
| Mean      | 20587.991193 | 21022.647802 | 20086.369442 | 20564.123041 | 1.897846e+10 |
| Std       | 23711.471881 | 24191.605193 | 23160.842989 | 23683.862047 | 2.071116e+10 |
| Min       | 178.102997  | 211.731003  | 171.509995  | 176.897003  | 5.914570e+06 |
| 25%       | 1278.034973 | 1287.570007 | 1265.265015 | 1275.320007 | 5.127045e+08 |
| 50%       | 9508.993164 | 9680.367188 | 9296.872070 | 9491.626953 | 1.505308e+10 |
| 75%       | 32941.566406 | 34125.681641 | 31323.410156 | 32844.773438 | 3.019101e+10 |
| Max       | 106146.265625 | 109114.882812 | 105291.734375 | 106147.296875 | 3.509679e+11 |

**Observations:** 
- The **mean closing price** is **$20,588**, but the **standard deviation** is **$23,711**, indicating large fluctuations.  
- The **minimum price** is **$178**, while the **maximum price** is **$106,146**, showing a wide price range.  
- The **median closing price** is **$9,508**, much lower than the **mean**, suggesting a **right-skewed distribution** with increasing prices over time.  
- Volume ranges from **$5.9 million** to **$350.97 billion**, indicating periods of extreme market activity.
 
**Ethereum Data**

| Statistic | Close       | High        | Low         | Open        | Volume         |
|-----------|------------|------------|------------|------------|---------------|
| **Count** | 2,658.000  | 2,658.000  | 2,658.000  | 2,658.000  | 2.658e+03     |
| **Mean**  | 1,520.832  | 1,561.608  | 1,474.429  | 1,520.105  | 1.327e+10     |
| **Std**   | 1,235.308  | 1,268.291  | 1,198.795  | 1,235.588  | 1.082e+10     |
| **Min**   | 84.308     | 85.343     | 82.830     | 84.280     | 6.217e+08     |
| **25%**   | 274.324    | 283.355    | 265.509    | 273.859    | 5.673e+09     |
| **50%**   | 1,455.794  | 1,524.832  | 1,415.312  | 1,449.179  | 1.079e+10     |
| **75%**   | 2,472.061  | 2,551.063  | 2,404.323  | 2,471.200  | 1.799e+10     |
| **Max**   | 4,812.087  | 4,891.705  | 4,718.039  | 4,810.071  | 9.245e+10     |

**Observations:**
- The **mean closing price** is **$1,520**, with a **standard deviation** of **$1,235**, showing significant volatility.
- Prices range from **$84** to **$4,812**, indicating a wide spread.
- The **median closing price** is **$1,455**, suggesting a **somewhat symmetric distribution**.
- Trading volume varies greatly, from **$621 million** to **$92.45 billion**, indicating occasional market surges.

**Solana Data**

| Statistic  | Close      | High       | Low        | Open       | Volume         |
|------------|------------|------------|------------|------------|----------------|
| **Count**  | 1775.0000  | 1775.0000  | 1775.0000  | 1775.0000  | 1.775000e+03   |
| **Mean**   | 72.475326  | 75.313932  | 69.523772  | 72.390474  | 1.697939e+09   |
| **Std**    | 71.119623  | 73.721297  | 68.421156  | 71.120493  | 2.239014e+09   |
| **Min**    | 0.515273   | 0.559759   | 0.505194   | 0.513391   | 6.520200e+05   |
| **25%**    | 18.246237  | 18.979635  | 17.388174  | 18.233952  | 2.604485e+08   |
| **50%**    | 35.556404  | 37.223209  | 33.735134  | 35.482449  | 1.069991e+09   |
| **75%**    | 136.751877 | 142.627663 | 131.819023 | 136.584198 | 2.442330e+09   |
| **Max**    | 261.869751 | 294.334961 | 253.187439 | 261.872437 | 3.317296e+10   |

**Observations:**

- The **mean closing price** is **$72.48**, with a **standard deviation of $71.12**, indicating significant price fluctuations.  
- The **minimum price** is **$0.51**, while the **maximum price** reaches **$261.87**, showing a wide range in value.  
- The **mean trading volume** is **$1.70 billion**, but the **standard deviation is high** (**$2.24 billion**), suggesting varying levels of trading activity.
- Volume ranges from **$652,020** to **$33.17 billion**, showing periods of both low and extreme market activity.

### Frequency Distribution

![image](https://github.com/user-attachments/assets/f23ecb4f-50f5-4284-8678-7e73b4b8968a)

![image](https://github.com/user-attachments/assets/423d0117-50ee-43a1-81b7-1b0e57c0e8b4)

![image](https://github.com/user-attachments/assets/b0c3f186-2b01-4888-b9ae-0f8f30d4a9eb)

- The close price shows a **right-skewed distribution**, indicating that lower prices were more frequent.
- There are **multiple peaks**, suggesting different historical price levels or market phases.  
- The trading volume is also **highly skewed**, with most transactions occurring at lower volumes.  
- There are **a few extreme values**, indicating periods of unusually high trading activity.
- The **steep decline** suggests that the majority of trading days had relatively low volume compared to the outliers.

### Correlation Martix

![image](https://github.com/user-attachments/assets/2c4751e0-4b73-4b84-a3ac-862efc82e0df)

### Closing Price Over Time

![image](https://github.com/user-attachments/assets/c09c5708-fcee-4eaa-99f5-b5bcfb6d1781)

![image](https://github.com/user-attachments/assets/102d2cef-d37a-4f97-a647-29acb91dd7d2)

**1️⃣ Bitcoin (BTC) Dominates Price Trends**
- **BTC (orange)** remains the highest-priced cryptocurrency.
- Surged past **$100,000 in 2024**, showing strong market confidence.
- **Historical peaks in 2017, 2021, and 2024** indicate repeated bull cycles.

**2️⃣ Ethereum (ETH) Shows Moderate Growth**
- **ETH (blue)** has a much lower price range compared to BTC.
- Peaked around **$5,000** in previous bull cycles.
- **Gradual upward trend**, showing solid adoption.

**3️⃣ Solana (SOL) Remains Relatively Lower in Price**
- **SOL (purple)** shows price spikes after 2021 but remains below BTC & ETH.

### Market Capitalization

![image](https://github.com/user-attachments/assets/a075b274-3c65-460d-a8c6-8daf3e31ab48)

**1️⃣ Bitcoin (BTC) Leads the Market**
- **BTC (orange)** has the highest market capitalization, peaking above **$2 trillion**.
- Significant **growth during bull runs** (2017, 2021, 2024).
- **Recent 2024 surge** suggests renewed investor confidence.

**2️⃣ Ethereum (ETH) Shows Strong Growth**
- **ETH (blue)** follows BTC but at a lower magnitude.
- Peaked around **$500 billion** in 2021 but remains **steadily increasing**.
- Indicates **strong network utility and adoption**.

**3️⃣ Solana (SOL) Gains Traction**
- **SOL (purple)** had a late start but saw significant growth post-2021.
- **Smaller market cap** compared to BTC & ETH but shows steady **adoption and resilience**.

**4️⃣ Market Cycles Are Clearly Visible**
- **Boom and bust cycles** are evident (2021 bull run, 2022 bear market).
- **Post-2023 recovery** shows renewed market interest.

### Trading Volume Analysis

![image](https://github.com/user-attachments/assets/276530bc-8a0d-4b2a-9f33-5579de69bc86)

1️⃣ Bitcoin (BTC) Dominates Trading Volume  
- **BTC (orange)** has the highest trading volume over time, especially during market peaks.  
- Major **spikes align with market cycles** (e.g., 2021 bull run).  

2️⃣ Ethereum (ETH) Has Consistently High Volume  
- **ETH (blue)** follows BTC’s trend but at a lower scale.  
- Shows **sustained liquidity**, indicating strong investor interest.  

3️⃣ Solana (SOL) Gained Traction Post-2020  
- **SOL (purple)** had minimal trading before 2020 but grew rapidly.  
- Lower volume than BTC & ETH, but **trading activity is increasing**.  

 4️⃣ Volume Spikes Correlate with Market Events  
- **2021:** Crypto bull run → **Highest trading activity ever recorded**.  
- **2022:** Market crash → **Sharp spikes, indicating panic selling**.  
- **2024:** Volume stabilizes but remains **volatile, especially for BTC**.

### Moving Averages (EMA)

![image](https://github.com/user-attachments/assets/8ee125cb-367c-4cc8-abb1-54eac94e8828)


### Volatility Analysis Using Rolling Standard Deviation

Volatility measures how much prices fluctuate over time. Higher volatility indicates higher risk but also higher potential returns.

![image](https://github.com/user-attachments/assets/4e21b0e6-8a7a-43b1-9f65-808c4adb9ce3)

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

## 7. Implementation

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

### **Seasonality Analysis:**  

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

## 7. Model Predictions

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
| **LSTM**           | 2147.02    | 2832.39    |
| **Random Forest**  | 4853.04    | 11581.81   |
| **XGBoost**        | 5498.91    | 12441.40   |
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

## 9. Model Deployment: 

  
