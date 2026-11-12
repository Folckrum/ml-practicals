import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# load and clean
df = pd.read_csv("uber.csv")
df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")
df = df.dropna()

# distance from coordinates (haversine formula)
lat1 = np.radians(df["pickup_latitude"])
lat2 = np.radians(df["dropoff_latitude"])
dlat = lat2 - lat1
dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
df["distance"] = 6371 * 2 * np.arcsin(np.sqrt(a))

# hour and year from the pickup time
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["hour"] = df["pickup_datetime"].dt.hour
df["year"] = df["pickup_datetime"].dt.year
df = df.drop(columns=["pickup_datetime"])

# remove wrong values
df = df[(df["fare_amount"] > 0) & (df["distance"] > 0)]
df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]

# remove outliers (IQR method)
for col in ["fare_amount", "distance"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

# correlation
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()

# train and test
X = df.drop(columns=["fare_amount"])
y = df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name)
    print("R2   :", r2_score(y_test, pred))
    print("RMSE :", np.sqrt(mean_squared_error(y_test, pred)))
    print()


evaluate("Linear Regression", LinearRegression())
evaluate("Random Forest", RandomForestRegressor(n_estimators=50, random_state=42))
