#!/usr/bin/env python
# coding: utf-8

# # Contoh 2 Classification

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:


df = pd.read_excel('book2.xlsx')


# In[3]:


df.describe()


# NB: Sebenarnya membuat klasifikasi dari data IPM dari variabel pembangunnya (UHH, Pengeluaran, dll) ini tidak boleh, karena ga guna buat apa capek2 bikin model, padahal bisa disortir pake excel. Tapi gapapalah buat contoh ae
# 
# <60 rendah
# 
# 60<= x <70 sedang
# 
# 70<= x <80 tinggi
# 
# =>80 sangat tinggi

# In[4]:


df.head()


# In[5]:


plt.figure(figsize=(12,8))
sns.distplot(df['IPM'])


# In[6]:


#membuat klasifikasi
bins = [0,60,70,80,100]


# In[7]:


data = df.copy() #df.copy agar data di df tidak berubah


# In[8]:


name = ['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']


# In[9]:


df['IPM'] = pd.cut(df.IPM, bins, labels=name)


# In[10]:


df.head()


# In[11]:


plt.figure(figsize=(5,5))
sns.countplot(x='IPM', data=df)


# eh ini gw gatau kenapa gabisa klasifikasi pake label nama wkwkwk, jadinya ditransform ke angka

# In[12]:


names = [0,1,2,3]


# In[13]:


data['IPM'] = pd.cut(data.IPM, bins, labels=names)


# In[14]:


data = data.drop('Prov_Kab_Kota', axis=1)


# In[15]:


data


# In[16]:


X = data.drop('IPM', axis=1).values
y = data['IPM'].values


# In[17]:


y


# In[18]:


plt.figure(figsize=(5,5))
sns.countplot(x='IPM', data=data)


# 0 = rendah
# 1 = sedang
# 2 = tinggi
# 3 = sangat tinggi

# One Hot Encoding
# 
# ngubah nilai klasifikasi tadi jadi bentuk array
# 
# rendah = 0 = [1. 0. 0. 0]
# 
# sedang = 1 = [0. 1. 0. 0]
# 
# dst
# 
# gw gatau alasan pake ini apa, pake ae
# soale klo gapake gabisa awokawok

# In[19]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)
y = le.fit_transform(y)
y = ohe.fit_transform(y.reshape(-1,1))
print(y)


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=221810445)


# In[21]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


# In[23]:


X_train.shape


# In[24]:


model = Sequential()

model.add(Dense(units=4,activation='relu'))
model.add(Dense(units=6,activation='relu'))
model.add(Dense(units=6,activation='relu'))
model.add(Dense(units=6,activation='relu'))
model.add(Dense(units=4,activation='softmax')) #klo binary classification diganti jadi sigmoid

# For a categorical classification problem
model.compile(loss='categorical_crossentropy', optimizer='adam')
# klo binary  classification diganti jadi binary_crossentropy


# In[25]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=500,
          verbose=1, 
          validation_data=(X_test, y_test))


# In[26]:


losses = pd.DataFrame(model.history.history)


# In[27]:


losses.plot()


# In[28]:


X_test


# In[29]:


predictions = model.predict_classes(X_test) #klo ada warning biarin ae


# In[30]:


predictions


# In[31]:


from sklearn.metrics import classification_report,confusion_matrix


# ini nge-invers one hot encoding tadi biar balik ke awal (0,1,2,3)

# In[32]:


inv_y = ohe.inverse_transform(y_test)
inv_y = le.inverse_transform(inv_y.astype(int).ravel())
inv_y


# In[33]:


print(classification_report(inv_y,predictions))


# akurasi 97% cuy
# yaiyalah orang yg jadi variabelnya, variabel pembangun IPM

# # Nyoba

# In[34]:


simeulue = data.drop('IPM', axis=1).iloc[0]
#iloc[0] untuk mengambil data pada baris pertama


# In[35]:


simeulue


# In[36]:


simeulue = simeulue.values.reshape(-1, 4) 


# In[37]:


simeulue


# In[38]:


simeulue = scaler.transform(simeulue) #minmaxscaler


# In[39]:


model.predict_classes(simeulue)


# In[40]:


df.head(1)


# 1 = sedang, bener lah ya
