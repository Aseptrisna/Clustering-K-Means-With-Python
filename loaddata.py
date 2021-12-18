import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


attendance_data = pd.read_csv("dataanalisa.csv")
attendance_data.head()
attendance_data.info()

attendance_data = attendance_data.drop(["No"], axis = 1)
attendance_data.head()
attendance_data.info()

attendance_data_x = attendance_data.iloc[:, 2:4]
attendance_data_x.head()
sns.scatterplot(x="total", y="label",
data=attendance_data, s=100, color="blue", alpha = 0.5)


x_array = np.array(attendance_data_x)
print(x_array)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
x_scaled

kmeans = KMeans(n_clusters = 5, random_state=123)
kmeans.fit(x_scaled)
print("K-means center")
print(kmeans.cluster_centers_)


print(kmeans.labels_)
attendance_data["kluster"] = kmeans.labels_
attendance_data.head()
attendance_data.info()
attendance_data.to_csv('resultdata17.csv');
fig, ax = plt.subplots()
sct = ax.scatter(x_scaled[:,1], x_scaled[:,0], s = 100,
c = attendance_data.kluster, marker = "o", alpha = 0.5) 
centers = kmeans.cluster_centers_
ax.scatter(centers[:,1], centers[:,0], c='blue', s=200, alpha=0.5);
plt.title("K-Means Clustering Results")
plt.xlabel("x")
plt.ylabel("y")
plt.show()