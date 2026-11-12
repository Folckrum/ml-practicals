import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("emails.csv")
X = df.drop(columns=["Email No.", "Prediction"])
y = df["Prediction"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    print()


evaluate("KNN", KNeighborsClassifier(n_neighbors=5))
evaluate("SVM", SVC(kernel="linear"))
