import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("sales_data_sample.csv", encoding="latin1")
X = df[["QUANTITYORDERED", "PRICEEACH", "SALES", "MSRP"]].dropna()
X = StandardScaler().fit_transform(X)

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.title("Elbow method")
plt.show()

k = int(input("Pick k from the elbow: "))
labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
print(pd.Series(labels).value_counts())
