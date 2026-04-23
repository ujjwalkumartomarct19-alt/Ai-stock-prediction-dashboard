import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ---------------- UI ----------------
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

st.title("🚀 Advanced AI Stock Prediction Dashboard")

# ---------------- INPUT ----------------
stock = st.sidebar.text_input("Enter Stock Symbol", "AAPL")

selected_model = st.sidebar.selectbox(
    "Select Model",
    ["Auto (Best Model)", "Linear Regression", "Random Forest", "XGBoost"]
)

# ---------------- LOAD DATA ----------------
data = yf.download(stock, period="1y")
data = data.reset_index()

# ---------------- FEATURE ENGINEERING ----------------
data["MA10"] = data["Close"].rolling(10).mean()
data["MA50"] = data["Close"].rolling(50).mean()
data["Return"] = data["Close"].pct_change()
data["Target"] = data["Close"].shift(-1)

data = data.dropna()

features = ["Open", "High", "Low", "Volume", "MA10", "MA50"]

data = data.dropna()

# --------- SAFETY CHECK ----------
if len(data) < 60:
    st.error("Not enough data to train model. Try another stock.")
    st.stop()

features = ["Open", "High", "Low", "Volume", "MA10", "MA50"]

X = data[features]
y = data["Target"]

# --------- TIME SERIES SPLIT ----------
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# ---------------- MODELS ----------------
lr_model = LinearRegression().fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
xgb_model = XGBRegressor(n_estimators=100).fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# ---------------- EVALUATION ----------------
lr_mae = mean_absolute_error(y_test, lr_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)

results = {
    "Linear Regression": lr_mae,
    "Random Forest": rf_mae,
    "XGBoost": xgb_mae
}

# ---------------- SELECT MODEL ----------------
if selected_model == "Auto (Best Model)":
    best_model_name = min(results, key=results.get)
else:
    best_model_name = selected_model

models = {
    "Linear Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

final_model = models[best_model_name]

# ---------------- PREDICTION ----------------
latest = X.iloc[-1].values.reshape(1, -1)

predicted_price = final_model.predict(latest)

if hasattr(predicted_price, "__len__"):
    predicted_price = predicted_price[0]

predicted_price = float(predicted_price)
current_price = data["Close"].iloc[-1]

# Handle if it's Series or invalid
if hasattr(current_price, "values"):
    current_price = current_price.values[0]

current_price = float(current_price)

# ---------------- CONFIDENCE SCORE ----------------
confidence = abs(predicted_price - current_price)

# ---------------- RECOMMENDATION ----------------
if predicted_price > current_price:
    recommendation = "BUY 📈"
elif predicted_price < current_price:
    recommendation = "SELL 📉"
else:
    recommendation = "HOLD ⚖️"

# ---------------- METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Current Price", f"${round(current_price,2)}")
col2.metric("Predicted Price", f"${round(predicted_price,2)}")
col3.metric("Model Used", best_model_name)
col4.metric("Confidence", f"{round(confidence,2)}")

# ---------------- RECOMMENDATION ----------------
st.subheader("📊 Recommendation")
st.success(recommendation)

# ---------------- MODEL COMPARISON (PLOTLY) ----------------
st.subheader("📉 Model Comparison")

fig = go.Figure(data=[
    go.Bar(x=list(results.keys()), y=list(results.values()))
])

fig.update_layout(title="Model Performance (MAE)", yaxis_title="Error")

st.plotly_chart(fig, use_container_width=True)

# ---------------- INTERACTIVE STOCK CHART ----------------
st.subheader("📈 Interactive Stock Chart")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=data["Date"], y=data["Close"],
    mode='lines', name='Close Price'
))

fig2.add_trace(go.Scatter(
    x=data["Date"], y=data["MA10"],
    mode='lines', name='MA10'
))

fig2.add_trace(go.Scatter(
    x=data["Date"], y=data["MA50"],
    mode='lines', name='MA50'
))

fig2.update_layout(title=f"{stock} Price Trend")

st.plotly_chart(fig2, use_container_width=True)

# ---------------- TOP 5 STOCK RECOMMENDATIONS ----------------
st.subheader("🏆 Top 5 Stock Recommendations")

stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

scores = []

for s in stocks:
    df = yf.download(s, period="3mo")
    df["MA10"] = df["Close"].rolling(10).mean()
    df = df.dropna()

    score = df["Close"].iloc[-1] - df["MA10"].iloc[-1]
    scores.append((s, score))

top5 = sorted(scores, key=lambda x: x[1], reverse=True)

for i, (s, sc) in enumerate(top5[:5], 1):
    st.write(f"{i}. {s} → Score: {round(sc,2)}")
