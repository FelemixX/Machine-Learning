#Naive Bayes Classifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Загрузка датасета ирисов
dataset = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Classification/master/IrisDataset.csv')
X = dataset.iloc[:,:4].values #Длина и ширина чашечки,длина и ширина лепестков
y = dataset['species'].values #Вид ириса
dataset.head(5) #выбираем первые пять ирисов для обучения
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) #Случайным образом перемешиваем выборку

sc = StandardScaler() #Нормализуем данные для обучения
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print ("\nТочность : ", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nМатрица путаницы\n",cm)

df = pd.DataFrame({'Реальные значения':y_test, 'Предугаданные значения':y_pred})
print("\n",df)

sns.set_style("whitegrid")
sns.FacetGrid(dataset, hue="species", height =6).map(plt.scatter,'sepal_length', 'petal_length').add_legend()