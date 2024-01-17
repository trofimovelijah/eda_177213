import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from explainerdashboard import *
from explainerdashboard.datasets import *

data_cl = pd.read_csv('abalone_cl.csv')
np.random.seed(42)

# матрица признаков
X = data_cl[['diameter', 'height', 'whole_weight', 'shell_weight']]
# вектор целевых (таргетированных) признаков
y = data_cl['rings']
# определение границ диапазонов
bins = [0, 4, 15, 29]
# определение меток для категорий
labels = ['неплодоносные', 'плодоносные', 'старые']
# группировка значений целевого признака по диапазонам
y_class = pd.cut(y, bins=bins, labels=labels, ordered=False)
# проверка результатов группировки
print(y_class.value_counts())

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

# преобразовываем категориальные значения целевого признака в числовые
y_test_encoded, y_test_categories = pd.factorize(y_test_full)
forest_explainer = ClassifierExplainer(gsb.best_estimator_, X_test_transformed, y_test_encoded)

db = ExplainerDashboard(forest_explainer)
db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)