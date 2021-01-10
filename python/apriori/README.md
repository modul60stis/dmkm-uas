# Apriori <img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/>

Library yang dibutuhkan `apyori`

## Import Library


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
```

## Import Data


```python
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
```


```python
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>shrimp</td>
      <td>almonds</td>
      <td>avocado</td>
      <td>vegetables mix</td>
      <td>green grapes</td>
      <td>whole weat flour</td>
      <td>yams</td>
      <td>cottage cheese</td>
      <td>energy drink</td>
      <td>tomato juice</td>
      <td>low fat yogurt</td>
      <td>green tea</td>
      <td>honey</td>
      <td>salad</td>
      <td>mineral water</td>
      <td>salmon</td>
      <td>antioxydant juice</td>
      <td>frozen smoothie</td>
      <td>spinach</td>
      <td>olive oil</td>
    </tr>
    <tr>
      <th>1</th>
      <td>burgers</td>
      <td>meatballs</td>
      <td>eggs</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chutney</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>turkey</td>
      <td>avocado</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mineral water</td>
      <td>milk</td>
      <td>energy bar</td>
      <td>whole wheat rice</td>
      <td>green tea</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7496</th>
      <td>butter</td>
      <td>light mayo</td>
      <td>fresh bread</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7497</th>
      <td>burgers</td>
      <td>frozen vegetables</td>
      <td>eggs</td>
      <td>french fries</td>
      <td>magazines</td>
      <td>green tea</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7498</th>
      <td>chicken</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7499</th>
      <td>escalope</td>
      <td>green tea</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7500</th>
      <td>eggs</td>
      <td>frozen smoothie</td>
      <td>yogurt cake</td>
      <td>low fat yogurt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>7501 rows × 20 columns</p>
</div>



## Rubah ke Bentuk Array


```python
transactions = []
for i in range(0, 7501): #7501 itu row data
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)]) #20 itu column data
transactions[1:3]
```




    [['burgers',
      'meatballs',
      'eggs',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan'],
     ['chutney',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan',
      'nan']]



## Buat Model


```python
rules = apriori(transactions = transactions, 
                min_support = 0.003, 
                min_confidence = 0.2, 
                min_lift = 3, 
                min_length = 2, 
                max_length = 2)
results = list(rules)
print(results)
```

    [RelationRecord(items=frozenset({'light cream', 'chicken'}), support=0.004532728969470737, ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'chicken'}), confidence=0.29059829059829057, lift=4.84395061728395)]), RelationRecord(items=frozenset({'escalope', 'mushroom cream sauce'}), support=0.005732568990801226, ordered_statistics=[OrderedStatistic(items_base=frozenset({'mushroom cream sauce'}), items_add=frozenset({'escalope'}), confidence=0.3006993006993007, lift=3.790832696715049)]), RelationRecord(items=frozenset({'escalope', 'pasta'}), support=0.005865884548726837, ordered_statistics=[OrderedStatistic(items_base=frozenset({'pasta'}), items_add=frozenset({'escalope'}), confidence=0.3728813559322034, lift=4.700811850163794)]), RelationRecord(items=frozenset({'fromage blanc', 'honey'}), support=0.003332888948140248, ordered_statistics=[OrderedStatistic(items_base=frozenset({'fromage blanc'}), items_add=frozenset({'honey'}), confidence=0.2450980392156863, lift=5.164270764485569)]), RelationRecord(items=frozenset({'ground beef', 'herb & pepper'}), support=0.015997866951073192, ordered_statistics=[OrderedStatistic(items_base=frozenset({'herb & pepper'}), items_add=frozenset({'ground beef'}), confidence=0.3234501347708895, lift=3.2919938411349285)]), RelationRecord(items=frozenset({'tomato sauce', 'ground beef'}), support=0.005332622317024397, ordered_statistics=[OrderedStatistic(items_base=frozenset({'tomato sauce'}), items_add=frozenset({'ground beef'}), confidence=0.3773584905660377, lift=3.840659481324083)]), RelationRecord(items=frozenset({'olive oil', 'light cream'}), support=0.003199573390214638, ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'olive oil'}), confidence=0.20512820512820515, lift=3.1147098515519573)]), RelationRecord(items=frozenset({'whole wheat pasta', 'olive oil'}), support=0.007998933475536596, ordered_statistics=[OrderedStatistic(items_base=frozenset({'whole wheat pasta'}), items_add=frozenset({'olive oil'}), confidence=0.2714932126696833, lift=4.122410097642296)]), RelationRecord(items=frozenset({'shrimp', 'pasta'}), support=0.005065991201173177, ordered_statistics=[OrderedStatistic(items_base=frozenset({'pasta'}), items_add=frozenset({'shrimp'}), confidence=0.3220338983050847, lift=4.506672147735896)])]
    


```python
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
resultsinDataFrame
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Left Hand Side</th>
      <th>Right Hand Side</th>
      <th>Support</th>
      <th>Confidence</th>
      <th>Lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>light cream</td>
      <td>chicken</td>
      <td>0.004533</td>
      <td>0.290598</td>
      <td>4.843951</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mushroom cream sauce</td>
      <td>escalope</td>
      <td>0.005733</td>
      <td>0.300699</td>
      <td>3.790833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pasta</td>
      <td>escalope</td>
      <td>0.005866</td>
      <td>0.372881</td>
      <td>4.700812</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fromage blanc</td>
      <td>honey</td>
      <td>0.003333</td>
      <td>0.245098</td>
      <td>5.164271</td>
    </tr>
    <tr>
      <th>4</th>
      <td>herb &amp; pepper</td>
      <td>ground beef</td>
      <td>0.015998</td>
      <td>0.323450</td>
      <td>3.291994</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tomato sauce</td>
      <td>ground beef</td>
      <td>0.005333</td>
      <td>0.377358</td>
      <td>3.840659</td>
    </tr>
    <tr>
      <th>6</th>
      <td>light cream</td>
      <td>olive oil</td>
      <td>0.003200</td>
      <td>0.205128</td>
      <td>3.114710</td>
    </tr>
    <tr>
      <th>7</th>
      <td>whole wheat pasta</td>
      <td>olive oil</td>
      <td>0.007999</td>
      <td>0.271493</td>
      <td>4.122410</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pasta</td>
      <td>shrimp</td>
      <td>0.005066</td>
      <td>0.322034</td>
      <td>4.506672</td>
    </tr>
  </tbody>
</table>
</div>



## Menggunakan Library `mlxtend`


```python
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

data = [
    ['Broccoli', 'Green Peppers', 'Corn'],
    ['Asparagus', 'Squash', 'Corn'],
    ['Corn', 'Tomatoes', 'Beans', 'Squash'],
    ['Green Peppers', 'Corn', 'Tomatoes', 'Beans'],
    ['Beans', 'Asparagus', 'Broccoli'],
    ['Squash', 'Asparagus', 'Beans', 'Tomatoes'],
    ['Tomatoes', 'Corn'],
    ['Broccoli', 'Tomatoes', 'Green Peppers'],
    ['Squash', 'Asparagus', 'Beans'],
    ['Beans', 'Corn'],
    ['Green Peppers', 'Broccoli', 'Beans', 'Squash'],
    ['Asparagus', 'Beans', 'Squash'],
    ['Squash', 'Corn', 'Asparagus', 'Beans'],
    ['Corn', 'Green Peppers', 'Tomatoes', 'Beans', 'Broccoli']
]
```

Struktur data yang digunakan harus seperti yang diatas

### Rubah Bentuk Data


```python
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Asparagus</th>
      <th>Beans</th>
      <th>Broccoli</th>
      <th>Corn</th>
      <th>Green Peppers</th>
      <th>Squash</th>
      <th>Tomatoes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



### Buat Model


```python
frequent_itemsets = apriori(df, min_support=0.30, use_colnames=True)
frequent_itemsets
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.428571</td>
      <td>(Asparagus)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.714286</td>
      <td>(Beans)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.357143</td>
      <td>(Broccoli)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.571429</td>
      <td>(Corn)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.357143</td>
      <td>(Green Peppers)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.500000</td>
      <td>(Squash)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.428571</td>
      <td>(Tomatoes)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.357143</td>
      <td>(Asparagus, Beans)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.357143</td>
      <td>(Asparagus, Squash)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.357143</td>
      <td>(Beans, Corn)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.428571</td>
      <td>(Beans, Squash)</td>
    </tr>
  </tbody>
</table>
</div>




```python
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
rules
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Asparagus)</td>
      <td>(Beans)</td>
      <td>0.428571</td>
      <td>0.714286</td>
      <td>0.357143</td>
      <td>0.833333</td>
      <td>1.166667</td>
      <td>0.051020</td>
      <td>1.714286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Asparagus)</td>
      <td>(Squash)</td>
      <td>0.428571</td>
      <td>0.500000</td>
      <td>0.357143</td>
      <td>0.833333</td>
      <td>1.666667</td>
      <td>0.142857</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Squash)</td>
      <td>(Asparagus)</td>
      <td>0.500000</td>
      <td>0.428571</td>
      <td>0.357143</td>
      <td>0.714286</td>
      <td>1.666667</td>
      <td>0.142857</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Squash)</td>
      <td>(Beans)</td>
      <td>0.500000</td>
      <td>0.714286</td>
      <td>0.428571</td>
      <td>0.857143</td>
      <td>1.200000</td>
      <td>0.071429</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>


