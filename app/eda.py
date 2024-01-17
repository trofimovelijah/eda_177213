#pip install polars
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
import scipy.stats as stats
import polars as pl

abalone_url = 'https://raw.githubusercontent.com/aiedu-courses/stepik_eda_and_dev_tools/main/datasets/abalone.csv'
data = pd.read_csv(abalone_url)

# Предобработка данных
## Приводим названия столбцов к нижнему регистру
data.columns = data.columns.str.lower()

## Заменяем пробелы на нижние подчеркивания
data.columns = data.columns.str.replace(' ', '_')

data.isnull().sum().sort_values(ascending=False)
data.dropna(subset=['shell_weight'], inplace=True)
data.interpolate(method='linear', inplace=True)

# Однофакторный анализ
## Определение и обработка уникальных значений
data['sex'].value_counts()
data['sex'] = data['sex'].str.upper()

# Построение графиков распределения по каждому из признаков
#### График плотности распределения длин раковин моллюска
plt.figure(figsize=(20,10))
data['length'].plot(
    kind='hist',
    density=True,
    alpha=0.8,
    bins=100,
    edgecolor='black'
                   )
data['length'].plot(
    kind='density',
    color='red'
                   )
plt.title('График распределения плотности длин раковин моллюска')
plt.xlim(0, 0.9)
plt.xlabel('Значение длины раковины Lenght, мм')
plt.ylabel('Суммарное число значений')
plt.grid(True)
plt.show()

#### График плотности распределения диаметра раковин моллюска
plt.figure(figsize=(20,10))
data['diameter'].plot(
    kind='hist',
    density=True,
    alpha=0.8,
    bins=100,
    edgecolor='black'
                   )
data['diameter'].plot(
    kind='density',
    color='red'
                   )
plt.title('График распределения диаметров раковин моллюска')
plt.xlim(0, 0.7)
plt.xlabel('Значение диаметра раковины Diameter, мм')
plt.ylabel('Суммарное число значений')
plt.grid(True)
plt.show()

#### График плотности распределения высоты раковин моллюска
plt.figure(figsize=(20,10))
data['height'].plot(
    kind='hist',
    density=True,
    alpha=0.8,
    bins=100,
    edgecolor='black'
                   )
data['height'].plot(
    kind='density',
    color='red'
                   )
plt.title('График распределения высоты раковин моллюска')
plt.xlim(-0.05, 0.3)
plt.xlabel('Значение высоты раковины Height, мм')
plt.ylabel('Суммарное число значений')
plt.grid(True)
plt.show()

##### Избавление от выбросов
data_cl = data[data['height'] < 0.4]

#### График плотности распределения массы целого моллюска
plt.figure(figsize=(20,10))
data_cl['whole_weight'].plot(
    kind='hist',
    density=True,
    alpha=0.8,
    bins=100,
    color='gray',
    edgecolor='black'
                   )
data_cl['whole_weight'].plot(
    kind='density',
    color='red'
                   )
plt.title('График распределения массы целого моллюска')
plt.xlim(-0.25, 3)
plt.xlabel('Значение массы целого моллюска whole_weight, гр')
plt.ylabel('Суммарное число значений')
plt.grid(True)
plt.show()

##### Избавление от выбросов
data_cl = data_cl[(data_cl['whole_weight'] <= 2.5)]

#### График плотности распределения массы очищенного моллюска
plt.figure(figsize=(20,10))
data_cl['shucked_weight'].plot(
    kind='hist',
    density=True,
    alpha=0.8,
    bins=100,
    color='gray',
    edgecolor='black'
                   )
data_cl['shucked_weight'].plot(
    kind='density',
    color='red'
                   )
plt.title('График распределения массы очищенного моллюска')
plt.xlim(-0.15, 1.5)
plt.xlabel('Значение массы мяса моллюска shucked_weight, гр')
plt.ylabel('Суммарное число значений')
plt.grid(True)
plt.show()

#### График плотности распределения массы потрохов моллюска
plt.figure(figsize=(20,10))
data_cl['viscera_weight'].plot(
    kind='hist',
    density=True,
    alpha=0.8,
    bins=100,
    color='gray',
    edgecolor='black'
                   )
data_cl['viscera_weight'].plot(
    kind='density',
    color='red'
                   )
plt.title('График распределения массы потрохов моллюска')
plt.xlim(-0.05, 0.8)
plt.xlabel('Значение массы потрохов моллюска viscera_weight, гр')
plt.ylabel('Суммарное число значений')
plt.grid(True)
plt.show()

#### График плотности распределения массы скорлупы моллюска
plt.figure(figsize=(20,10))
data_cl['shell_weight'].plot(
    kind='hist',
    density=True,
    alpha=0.8,
    bins=100,
    color='gray',
    edgecolor='black'
                   )
data_cl['shell_weight'].plot(
    kind='density',
    color='red'
                   )
plt.title('График распределения массы раковины моллюска')
plt.xlim(-0.25, 1.2)
plt.xlabel('Значение массы скорлупы моллюска shell_weight, гр')
plt.ylabel('Суммарное число значений')
plt.grid(True)
plt.show()

##### Избавление от выбросов
data_cl = data_cl[(data_cl['viscera_weight'] < 0.625)]


###### общее число различных колец
len_rings = len(data_cl['rings'].unique())
len_rings

#### График плотности распределения колец раковин
plt.figure(figsize=(20,10))
data_cl['rings'].plot(
    kind='hist',
    density=True,
    alpha=0.8,
    bins=len_rings,
    color='green',
    edgecolor='black'
                   )
data_cl['rings'].plot(
    kind='density',
    color='red'
                   )
plt.title('График распределения колец на раковине')
plt.xlim(0, 29)
plt.xlabel('Число колец раковины rings')
plt.ylabel('Суммарное число значений')
plt.grid(True)
plt.show()

## Анализ целевой переменной
print(f'По результату очистки данных в ходе ИАД датасет составляет {round(100 * data_cl.shape[0] / data.shape[0], 2)}% от первоначального')
print(f'Число строк в датасете {data_cl.shape[0]}, число признаков {data_cl.shape[1]}')

#### Проверка балансировки таргетов для целевого признака
data_cl['rings'].value_counts()
data_cl = data_cl[data_cl['rings'] <= 23]

## Построение матриц корреляций
corr = data_cl.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(20,10))
sns.heatmap(corr, mask=mask, cmap="Reds", annot=True)
plt.title('матрица корреляции Пирсона')
plt.show()

corr = data_cl.corr(method='spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(20,10))
sns.heatmap(corr, mask=mask, cmap="PuBuGn", annot=True)
plt.title('матрица корреляции Спирмена')
plt.show()

### Тест $X^2$
# создаем таблицу сопряженности для категориальных признаков
contingency_table = pd.crosstab(data_cl['sex'], data_cl['rings'])

chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f'статистика теста   {chi2}\n\np-значение   {p_value}\n\nчисло степеней свободы   {dof}')
print()
print(f'ожидаемые частоты {expected}')

### ANOVA
grouped_data = data_cl.groupby('sex')['rings'].apply(list)
f_statistic, p_value = stats.f_oneway(*grouped_data)

print(f'F-статистика {f_statistic}\np-value {p_value}')
print('\nВывод:')
if p_value >= 0.05:
    print('Нулевая гипотеза (о равенстве средних значений) верна')
else:
    print('Нулевая гипотеза (о равенстве средних значений) отвергается')



grouped_data = data_cl.groupby('sex')['diameter'].apply(list)
f_statistic, p_value = stats.f_oneway(*grouped_data)

print(f'F-статистика {f_statistic}\np-value {p_value}')
print('\nВывод:')
if p_value >= 0.05:
    print('Нулевая гипотеза (о равенстве средних значений) верна')
else:
    print('Нулевая гипотеза (о равенстве средних значений) отвергается')

data_cl.to_csv('abalone_cl.csv', index=False)