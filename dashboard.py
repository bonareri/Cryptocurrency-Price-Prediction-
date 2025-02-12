import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv", parse_dates=["ds"])
    return df

df = load_data()

# Page title
st.title("📊 Bitcoin Analysis Dashboard")

# Sidebar filters
st.sidebar.header("Filter Data")
date_range = st.sidebar.date_input("Select Date Range", [df["ds"].min(), df["ds"].max()])
filtered_df = df[(df["ds"] >= pd.to_datetime(date_range[0])) & (df["ds"] <= pd.to_datetime(date_range[1]))]

# Bitcoin Price Forecast
st.subheader("📈 Bitcoin Price Forecast")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=filtered_df["ds"], y=filtered_df["yhat"], mode='lines', name='Predicted Price'))
fig1.add_trace(go.Scatter(x=filtered_df["ds"], y=filtered_df["yhat_lower"], mode='lines', name='Lower Bound', line=dict(dash='dot')))
fig1.add_trace(go.Scatter(x=filtered_df["ds"], y=filtered_df["yhat_upper"], mode='lines', name='Upper Bound', line=dict(dash='dot')))
st.plotly_chart(fig1)

# RSI Indicator
st.subheader("📊 Relative Strength Index (RSI)")
fig2 = px.line(filtered_df, x="ds", y="RSI", title="RSI Over Time", labels={"RSI": "Relative Strength Index"})
st.plotly_chart(fig2)

# Moving Averages
st.subheader("📊 Moving Averages (SMA 50 & SMA 100)")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=filtered_df["ds"], y=filtered_df["SMA_50"], mode='lines', name='SMA 50'))
fig3.add_trace(go.Scatter(x=filtered_df["ds"], y=filtered_df["SMA_100"], mode='lines', name='SMA 100'))
st.plotly_chart(fig3)

# MACD Analysis
st.subheader("📊 MACD & Signal Line")
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=filtered_df["ds"], y=filtered_df["MACD"], mode='lines', name='MACD'))
fig4.add_trace(go.Scatter(x=filtered_df["ds"], y=filtered_df["Signal_Line"], mode='lines', name='Signal Line'))
st.plotly_chart(fig4)

# Volatility & Risk Metrics
st.subheader("📉 Rolling Volatility & Sharpe Ratio")
col1, col2 = st.columns(2)
with col1:
    fig5 = px.line(filtered_df, x="ds", y="Rolling_Volatility", title="Rolling Volatility")
    st.plotly_chart(fig5)
with col2:
    fig6 = px.line(filtered_df, x="ds", y="Sharpe_Ratio", title="Sharpe Ratio")
    st.plotly_chart(fig6)

# Conclusion
st.markdown("### Key Insights")
st.write("- The predicted Bitcoin price shows a general trend along with confidence intervals.")
st.write("- RSI fluctuations indicate overbought/oversold conditions.")
st.write("- SMA 50 vs. SMA 100 shows short-term vs. long-term momentum.")
st.write("- MACD & Signal Line crossover helps identify trend reversals.")
st.write("- Rolling volatility & Sharpe ratio provide insights into risk and returns.")
