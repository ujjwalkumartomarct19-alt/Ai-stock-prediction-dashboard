#!/usr/bin/env python
# coding: utf-8

# # importing lib-

# In[19]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import yfinance as yf
import pandas as pd



# # Get Real-Time Stock Data

# In[20]:


stock = "AAPL"   # you can change

data = yf.download(stock, period="1y")
data = data.reset_index()

print(data.head())


# # feacutre engineering

# In[21]:


data["MA10"] = data["Close"].rolling(10).mean()
data["MA50"] = data["Close"].rolling(50).mean()
data["Return"] = data["Close"].pct_change()

data["Target"] = data["Close"].shift(-1)

data = data.dropna()
#data.head()


# # prepare data

# In[22]:


features = ["Open", "High", "Low", "Volume", "MA10", "MA50"]

X = data[features]
y = data["Target"]


# # tran/test and split

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)


# # train xgboost

# In[24]:


# Model 1: Random Forest
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10)
rf_model.fit(X_train, y_train)

# Model 2: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# In[25]:


from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)

xgb_model.fit(X_train, y_train)


# # prediction

# In[26]:


rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)


# # Compare using MAE + R²

# In[31]:


from sklearn.metrics import mean_absolute_error, r2_score

results = {
    "Linear Regression": {
        "MAE": mean_absolute_error(y_test, lr_pred),
        "R2": r2_score(y_test, lr_pred)
    },
    "Random Forest": {
        "MAE": mean_absolute_error(y_test, rf_pred),
        "R2": r2_score(y_test, rf_pred)
    },
    "XGBoost": {
        "MAE": mean_absolute_error(y_test, xgb_pred),
        "R2": r2_score(y_test, xgb_pred)
    }
}

for model, metrics in results.items():
    print(f"{model} → MAE: {metrics['MAE']:.2f}, R2: {metrics['R2']:.2f}")


# # SELECT BEST MODEL

# In[32]:


best_model_name = min(results, key=lambda x: results[x]["MAE"])
print("Best Model:", best_model_name)


# In[39]:


models = {
    "Linear Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

final_model = models[best_model_name]

latest = X.iloc[-1].values.reshape(1, -1)

predicted_price = final_model.predict(latest)[0]
current_price = data["Close"].iloc[-1]
print("Current Price:",current_price)
print("Predicted Price:",predicted_price)


# In[44]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(data["Date"], data["Close"], label="Price")
plt.plot(data["Date"], data["MA10"], label="MA10")
plt.plot(data["Date"], data["MA50"], label="MA50")

plt.legend()
plt.title(f"{stock} Stock Trend")
plt.show()


# In[ ]:





# # VISUAL COMPARISION

# In[35]:


import matplotlib.pyplot as plt

models = list(results.keys())
mae_values = [results[m]["MAE"] for m in models]

plt.bar(models, mae_values)
plt.title("Model Comparison (MAE)")
plt.show()


# In[37]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(y_test.values, label="Actual")
plt.plot(rf_pred, label="Random Forest")
plt.plot(lr_pred, label="Linear Regression")
plt.plot(xgb_pred,label="XGBoost regressor")
plt.legend()
plt.title("Model Comparison")
plt.show()


# In[40]:


from textblob import TextBlob

news = [
    "Market is growing strongly",
    "Investors are optimistic"
]

scores = [TextBlob(n).sentiment.polarity for n in news]
sentiment = sum(scores)/len(scores)

print("Sentiment:", sentiment)


# In[46]:


top5 = data.sort_values(by="Return", ascending=False).head(5)
print(top5[["Date", "Close", "Return"]])


# In[42]:


if predicted_price > current_price:
    recommendation = "BUY 📈"
elif predicted_price < current_price:
    recommendation = "SELL 📉"
else:
    recommendation = "HOLD ⚖️"

print("Recommendation:", recommendation)


# In[ ]:





# In[ ]:




