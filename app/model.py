import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import shap
from sklearn.inspection import permutation_importance as pi
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data_cl = pd.read_csv('abalone_cl.csv')
data_cl.head()

# установка начального значения генератора случайных чисел в ходе обучения
np.random.seed(42)

# матрица признаков
X = data_cl[['diameter', 'height', 'whole_weight', 'shell_weight']]
# вектор целевых (таргетированных) признаков
y = data_cl['rings']

### Разбиение на возрастные группы
# определение границ диапазонов
bins = [0, 4, 15, 29]
# определение меток для категорий
labels = ['неплодоносные', 'плодоносные', 'старые']
# группировка значений целевого признака по диапазонам
y_class = pd.cut(y, bins=bins, labels=labels, ordered=False)
# проверка результатов группировки
print(y_class.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.25)

### Метод наивного Байеса (Naive Bayes)
nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

nb_accuracy_score = accuracy_score(y_test.values.ravel(), y_pred)
nb_precision_score = precision_score(y_test.values.ravel(), y_pred, average='micro')
nb_recall_score = recall_score(y_test.values.ravel(), y_pred, average='micro')

print("NB Метрика качества accurancy_score", nb_accuracy_score)
print("NB Метрика качества precision_score", nb_precision_score)
print("NB Метрика качества recall_score", nb_recall_score)

### Метод ближайшего соседа (Nearest Neighbor)
knn = KNeighborsClassifier()

knn.fit(X_train, y_train.values.ravel())

y_pred_knn = knn.predict(X_test)

knn_accuracy_score = accuracy_score(y_test.values.ravel(), y_pred_knn)
knn_precision_score = precision_score(y_test, y_pred_knn, average='micro')
knn_recall_score = recall_score(y_test, y_pred_knn, average='micro')

print("KNN Метрика качества accurancy_score", knn_accuracy_score)
print("KNN Метрика качества precision_score", knn_precision_score)
print("KNN Метрика качества recall_score", knn_recall_score)

### Метод дерева решений
dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

y_pred_dtc = dtc.predict(X_test)

dtc_accuracy_score = accuracy_score(y_test.values.ravel(), y_pred_dtc)
dtc_precision_score = precision_score(y_test, y_pred_dtc, average='micro')
dtc_recall_score = recall_score(y_test, y_pred_dtc, average='micro')

print("DTC Метрика качества accurancy_score", dtc_accuracy_score)
print("DTC Метрика качества precision_score", dtc_precision_score)
print("DTC Метрика качества recall_score", dtc_recall_score)

## Подбор гиперпараметров каждой из моделей
### NB
model_nb = GaussianNB()
param_grid = {
    'var_smoothing': np.logspace(-10, 0, num=100)
             }

gs = GridSearchCV(model_nb, param_grid, cv=3, scoring='accuracy', verbose=1)

gs.fit(X_train, y_train)

print()
print(f'NB Лучшее значение гиперпараметров {gs.best_params_}')
print(f'NB Лучшее значение метрики качества при указанных значениях гиперпараметров {gs.best_score_}')
print()
pred = gs.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('NB Значение метрики качества на тестовой выборке:', accuracy)
print()

### KNN
model_knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2, 3, 4]
             }

gs = GridSearchCV(model_knn, param_grid, cv=5, scoring='accuracy', verbose=1)

gs.fit(X_train, y_train)

print()
print(f'KNN Лучшее значение гиперпараметров {gs.best_params_}')
print(f'KNN Лучшее значение метрики качества при указанных значениях гиперпараметров {gs.best_score_}')
print()
pred = gs.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('KNN Значение метрики качества на тестовой выборке:', accuracy)
print()

### DTC
model_dt = DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
             }

gs = GridSearchCV(model_dt, param_grid, cv=5, scoring='accuracy', verbose=1)

gs.fit(X_train, y_train)

print()
print(f'DTC Лучшее значение гиперпараметров {gs.best_params_}')
print(f'DTC Лучшее значение метрики качества при указанных значениях гиперпараметров {gs.best_score_}')
print()
pred = gs.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('DTC Значение метрики качества на тестовой выборке:', accuracy)
print()

## Добавление категориальных признаков в лучшую модель
X_full = data_cl.drop('rings', axis=1)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_class, test_size=0.25)
X_full.info()
# выделяем категориальный признак
categorical = ['sex']
# количественные признаки
numeric_features = ['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']
ct = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical),
    ('scaling', MinMaxScaler(), numeric_features)
])
X_train_transformed = ct.fit_transform(X_train_full)
X_test_transformed = ct.transform(X_test_full)
new_features = list(ct.named_transformers_['ohe'].get_feature_names_out())
new_features.extend(numeric_features)

print('Всего количественных признаков:', new_features)
X_train_transformed = pd.DataFrame(X_train_transformed, columns=new_features)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=new_features)

X_train_transformed.head()

### Построение модели с количественными признаками
model_dt = DecisionTreeClassifier()

params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
             }

gsb = GridSearchCV(model_dt, params, cv=5, scoring='accuracy', verbose=1)

gsb.fit(X_train_transformed, y_train_full)

print()
print(f'Лучшее значение гиперпараметров {gsb.best_params_}')
print(f'Лучшее значение метрики качества при указанных значениях гиперпараметров {gsb.best_score_}')
print()
# Вычисляем точность на тестовой выборке
pred_best = gsb.best_estimator_.predict(X_test_transformed)
accuracy_best = accuracy_score(y_test_full, pred_best)
print('Значение метрики качества на тестовой выборке:', accuracy_best)
print()

### Permutation Importance
X_train_transformed.shape
plt.figure(figsize=(20,10))
result = pi(
    gsb.best_estimator_,
    X_train_transformed,
    y_train_full,
    n_repeats=10,
    random_state=0
                               )
sorted_idx = result.importances_mean.argsort()
plt.barh(
    X_train_transformed.columns[sorted_idx],
    result.importances_mean[sorted_idx]
        )
plt.xlabel("Permutation Importance")
plt.show()

### Shapley values
X_train_transformed.shape
model_best_dt = gsb.best_estimator_
model_best_dt.fit(X_train_transformed, y_train_full)
explainer = shap.KernelExplainer(model_best_dt.predict_proba, X_train_transformed)
shap_values = explainer.shap_values(X_test_transformed.iloc[:150,:])
shap.summary_plot(shap_values, X_test_transformed.iloc[:150,:])
plt.show()