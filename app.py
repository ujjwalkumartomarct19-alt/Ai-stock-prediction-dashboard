import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import streamlit as st
# other imports...

# 👇 ADD CSS HERE (ONLY ONCE)
st.markdown("""
<style>

/* 🔥 Animated Background */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c1c);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* ✨ Fade-in Animation */
.fade-in {
    animation: fadeIn 1.5s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* 💎 Card Style */
.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    transition: 0.3s;
}

.card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(0,255,255,0.5);
}

/* ⚡ Glow Text */
.glow {
    font-size: 24px;
    font-weight: bold;
    color: #00ffcc;
    text-align: center;
    animation: glow 1.5s infinite alternate;
}

@keyframes glow {
    from {text-shadow: 0 0 10px #00ffcc;}
    to {text-shadow: 0 0 25px #00ffcc;}
}

/* 🚀 Button Style */
.stButton>button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.1);
    box-shadow: 0 0 15px #00c6ff;
}

</style>
""", unsafe_allow_html=True)

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
import plotly.graph_objects as go

st.subheader("📈 Interactive Stock Chart")

fig2 = go.Figure()

# Close Price (bright cyan)
fig2.add_trace(go.Scatter(
    x=data["Date"], 
    y=data["Close"],
    mode='lines',
    name='Close Price',
    line=dict(color='cyan', width=2)
))

# MA10 (yellow)
fig2.add_trace(go.Scatter(
    x=data["Date"], 
    y=data["MA10"],
    mode='lines',
    name='MA10',
    line=dict(color='yellow', width=2)
))

# MA50 (red)
fig2.add_trace(go.Scatter(
    x=data["Date"], 
    y=data["MA50"],
    mode='lines',
    name='MA50',
    line=dict(color='red', width=2)
))

# 🔥 Important: Dark theme layout
fig2.update_layout(
    template="plotly_dark",
    title=f"{stock} Price Trend",
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white')
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- TOP 5 STOCK RECOMMENDATIONS ----------------
st.subheader("🏆 Top 5 Stock Recommendations")

stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

scores = []

for s in stocks:
    df = yf.download(s, period="3mo")

    if df.empty:
        continue

    df["MA10"] = df["Close"].rolling(10).mean()
    df = df.dropna()

    if len(df) == 0:
        continue

    close_val = df["Close"].iloc[-1]
    ma_val = df["MA10"].iloc[-1]

    # Fix Series issue
    if hasattr(close_val, "values"):
        close_val = close_val.values[0]

    if hasattr(ma_val, "values"):
        ma_val = ma_val.values[0]

    score = float(close_val) - float(ma_val)

    scores.append((s, score))

# Sort safely
top5 = sorted(scores, key=lambda x: float(x[1]), reverse=True)

for i, (s, sc) in enumerate(top5[:5], 1):
    st.write(f"{i}. {s} → Score: {round(sc,2)}")
