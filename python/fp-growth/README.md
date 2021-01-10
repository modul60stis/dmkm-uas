# Fp-Growth <img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/>

## Import Data dan Library


```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

dataset = [
    ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
    ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
    ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']
]
```

Datasets harus berbentuk seperti diatas

## Rubah Struktur Data


```python
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Corn</th>
      <th>Dill</th>
      <th>Eggs</th>
      <th>Ice cream</th>
      <th>Kidney Beans</th>
      <th>Milk</th>
      <th>Nutmeg</th>
      <th>Onion</th>
      <th>Unicorn</th>
      <th>Yogurt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



## Buat Model


```python
frequent_itemsets = fpgrowth(df, use_colnames=True)
frequent_itemsets
```




<div>
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
      <td>1.0</td>
      <td>(Kidney Beans)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8</td>
      <td>(Eggs)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.6</td>
      <td>(Yogurt)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.6</td>
      <td>(Onion)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6</td>
      <td>(Milk)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.8</td>
      <td>(Eggs, Kidney Beans)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.6</td>
      <td>(Kidney Beans, Yogurt)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.6</td>
      <td>(Eggs, Onion)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.6</td>
      <td>(Onion, Kidney Beans)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.6</td>
      <td>(Eggs, Onion, Kidney Beans)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.6</td>
      <td>(Kidney Beans, Milk)</td>
    </tr>
  </tbody>
</table>
</div>




```python
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
rules
```




<div>
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
      <td>(Eggs)</td>
      <td>(Kidney Beans)</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>0.8</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Kidney Beans)</td>
      <td>(Eggs)</td>
      <td>1.0</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>0.80</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Yogurt)</td>
      <td>(Kidney Beans)</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Eggs)</td>
      <td>(Onion)</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>1.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Onion)</td>
      <td>(Eggs)</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(Onion)</td>
      <td>(Kidney Beans)</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(Eggs, Onion)</td>
      <td>(Kidney Beans)</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(Eggs, Kidney Beans)</td>
      <td>(Onion)</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>1.6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(Onion, Kidney Beans)</td>
      <td>(Eggs)</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(Eggs)</td>
      <td>(Onion, Kidney Beans)</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>1.6</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(Onion)</td>
      <td>(Eggs, Kidney Beans)</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(Milk)</td>
      <td>(Kidney Beans)</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>inf</td>
    </tr>
  </tbody>
</table>
</div>


