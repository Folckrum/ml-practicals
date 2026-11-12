import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

df = pd.read_csv("Churn_Modelling.csv")
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

X = df.drop(columns=["Exited"]).astype("float32")
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_features = X_train.shape[1]


def train_and_test(model):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    pred = (model.predict(X_test) > 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    print()


print("Basic model")
basic = Sequential([
    Input(shape=(n_features,)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid"),
])
train_and_test(basic)

# improvement: more neurons + dropout to reduce overfitting
print("Improved model")
improved = Sequential([
    Input(shape=(n_features,)),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])
train_and_test(improved)
