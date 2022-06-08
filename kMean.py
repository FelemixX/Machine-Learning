import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def prepare_housing_dataset(h, training_percent):
    """
    Подготавливает данные из датасета housing
    :param h pandas.DataFrame
    :param training_percent percent of training data
    """
    length = h.shape[1] #сделать одномерный массив 
    var_quantity = 5
    train_quantity = int(np.round(length * (training_percent / 100)))
    h_prepared = {"train_data": np.empty(shape=[train_quantity, var_quantity], dtype=int),
                  "train_target": np.empty(shape=train_quantity, dtype=int),
                  "test_data": np.empty(shape=[length - train_quantity, var_quantity], dtype=int),
                  "test_target": np.empty(shape=length - train_quantity, dtype=int),
                  "names": np.array([
                        "income",
                        "rooms_per_house",
                        "near_ocean",
                        "near_bay",
                        "1h_ocean",
                        "new_house",
                        "people_per_house",
                        "latitude",
                        "longitude",
                  ]),
                  "target_names": np.array(
                      ["<50000$", "50 000-150 000 $", "150 000-250 000 $", "250 000-350 000 $", "350 000-450 000 $",
                       "450 000-550 000 $"])
                  }
    for i in range(train_quantity):
        row = h[i]
        h_prepared["train_target"][i] = row["median_house_value"]
        h_prepared["train_data"][i] = np.ceil(row.drop("median_house_value")*10)
    for i in range(length - train_quantity):
        row = h[i]
        h_prepared["test_target"][i] = row["median_house_value"]
        h_prepared["test_data"][i] = np.ceil(row.drop("median_house_value")*10)
    return h_prepared
  
  sns.set()
path = "dataset/"
#housing = pd.read_csv()
housing = pd.read_csv(path + "housing_new_flipped.csv", index_col=0)
housing.columns = housing.columns.astype(int)
prepared = prepare_housing_dataset(housing, 100) #отправляем в работу датасет housing и тренируем на 100% представленных данных
print("Selected training clusters: ", [np.min(prepared["train_target"]), np.max(prepared["train_target"])]) #вывод выбранного кластера 
kmeans = KMeans(n_clusters=6, random_state=45) #встроенный метод кластеризации
clusters = kmeans.fit_predict(prepared['train_data']) #поместить данные в модель для тренировки

from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(6):
    mask = (clusters == i)
    labels[mask] = mode(prepared['train_target'][mask])[0]
    print(labels)
from sklearn.metrics import accuracy_score, confusion_matrix

print("Selected clusters: ", [np.min(labels), np.max(labels)]) #кластеры которые пошли на бой
print("Accuracy is: ", accuracy_score(prepared['train_target'], labels))
mat = confusion_matrix(prepared["train_target"], labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=prepared["target_names"], yticklabels=prepared["target_names"])
plt.xlabel('true median house value')
plt.ylabel('predicted median house value')
plt.show()
