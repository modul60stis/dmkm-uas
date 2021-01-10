#!/usr/bin/env python
# coding: utf-8

# # Fp-Growth

# ## Import Data dan Library

# In[1]:


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


# Datasets harus berbentuk seperti diatas

# ## Rubah Struktur Data

# In[2]:


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
df


# ## Buat Model

# In[3]:


frequent_itemsets = fpgrowth(df, use_colnames=True)
frequent_itemsets


# In[4]:


rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
rules

