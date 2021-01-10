#!/usr/bin/env python
# coding: utf-8

# # Apriori

# Library yang dibutuhkan `apyori`

# ## Import Library

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


# ## Import Data

# In[2]:


dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)


# In[3]:


dataset


# ## Rubah ke Bentuk Array

# In[4]:


transactions = []
for i in range(0, 7501): #7501 itu row data
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)]) #20 itu column data
transactions[1:3]


# ## Buat Model

# In[5]:


rules = apriori(transactions = transactions, 
                min_support = 0.003, 
                min_confidence = 0.2, 
                min_lift = 3, 
                min_length = 2, 
                max_length = 2)
results = list(rules)
print(results)


# In[6]:


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
resultsinDataFrame


# ## Menggunakan Library `mlxtend`

# In[7]:


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


# Struktur data yang digunakan harus seperti yang diatas

# ### Rubah Bentuk Data

# In[8]:


te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)
df


# ### Buat Model

# In[9]:


frequent_itemsets = apriori(df, min_support=0.30, use_colnames=True)
frequent_itemsets


# In[10]:


rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
rules

