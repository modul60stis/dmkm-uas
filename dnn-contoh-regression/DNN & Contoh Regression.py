#!/usr/bin/env python
# coding: utf-8

# # Neural Network

# ### Library yang dibutuhkan

# Di sini menggunakan Framework Tensorflow
# Library yang akan digunakan adalah **pandas, numpy, matplotlib, seaborn, sklearn, dan tensorflow**. Jika belum diinstall, silahkan diinstal dahulu dengan mengetikkan `pip install nama-library` pada anaconda prompt.
# 
# note: Library tensorflow cukup besar sizenya, sekitar 300MB.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


#   

# ### Sedikit penjelasan mengenai Deep Neural Network

# **Neural network** adalah sebuah model yang berisikan node atau neuron atau perceptron yang saling terhubung dengan dengan node lain melalui koneksi yang disebut penimbang atau weight. Neural network terbagi menjadi tiga bagian utama:
# Input Layer
# Hidden Layer
# Output Layer
# 
# Deep Neural Network (DNN) adalah perluasan dari metode neural network. Perbedaannya adalah dalam DNN mempunyai **lebih dari dua hidden layer**. 
# <img src="figure/dnn.png">

# ### Activation Function

# Activation function berfungsi untuk menentukan apakah neuron tersebut harus aktif atau tidak.
# Activation yang biasanya digunakan:
# 
#     sigmoid = untuk binary classification
#     tanh    = untuk binary classification
#     softmax = untuk categorical classification
#     ReLU
#    lainnya bisa dilihat di `https://en.wikipedia.org/wiki/Activation_function`

#  

# ### Loss Function 

# Loss Functioin berfungsi untuk mengukur seberapa besar nilai error yang dihasilkan dari output terhadap nilai aslinya.
# Loss function yang biasa digunakan adalah `binary_corssentropy`, `categorical_crossentropy` -> untuk klasifikasi. dan `mse` untuk regresi.

# ### Backpropagation

# Mudahnya backpropagation adalah suatu cara untuk menyesuaikan penimbang dan bias yang dihasilkan untuk meminimalisir error pada output.
# <img src="figure/nn.png">
# Caranya adalah
# 1. Menghitung nilai error menggunakan loss function
# 2. Hitung gradien dari loss funtion tersebut
# <img src="figure/loss_opt.png">
# <img src="figure/loss_opt2.png">
# 3. Update parameter bias dan penimbang dari hasil gradien yang didapatkan
# 
# nb: belum terlalu paham dengan backpropagation
# 
# Optimizer yang biasa digunakan adalah `adam`, `rmsprop`
# 
# Penjelasannya juga bisa dilihat di sini `https://medium.com/@samuelsena/pengenalan-deep-learning-part-3-backpropagation-algorithm-720be9a5fbb8` dan bisa juga search sendiri ya

# # Contoh 1 Regression

# Import Data

# In[2]:


df = pd.read_csv('california_housing_sklearn.csv')


# In[3]:


df.head()


# Kita akan meregresikan SalePrice (Y) dengan variable independent MedInc, HouseAge, AveRooms, AveBedrms, Population, dan AveOccup.

# In[4]:


df.isnull().sum()


# In[5]:


df.describe().transpose()


# #### Distribution Plot dari Sale Price

# In[6]:


plt.figure(figsize=(10,8))
sns.distplot(df['SalePrice'])


# #### Korelasi

# In[7]:


df.corr()


# In[8]:


df.corr()['SalePrice'].sort_values()


# #### Scatter Plot

# In[9]:


sns.scatterplot(x='AveRooms', y='SalePrice', data=df)


# In[10]:


sns.scatterplot(x='Longitude', y='Latitude', data=df, edgecolor=None, alpha=0.2, hue='SalePrice')


# ### Creating a model

# In[11]:


data=df.drop(['Longitude', 'Latitude'], axis=1) #Membuang variabel Longitude dan Latitude #axis=1 untuk kolom, axis=0 untuk baris


# In[12]:


data.head()


# In[13]:


X = data.drop('SalePrice', axis=1).values
y = data['SalePrice']


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


#Split data menjadi data training dan testing

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=221810445)


# #### Normalizing (MinMaxScaler)

# In[16]:


from sklearn.preprocessing import MinMaxScaler


# In[17]:


scaler = MinMaxScaler()


# In[18]:


X_train= scaler.fit_transform(X_train)


# In[19]:


X_test = scaler.transform(X_test)


# In[20]:


X_train.shape #ada 6 kolom dan 14449 baris


# ### Creating Model

# In[21]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


# In[22]:


model = Sequential()

model.add(Dense(6,activation='relu')) #input layer
model.add(Dense(6,activation='relu')) #hidden layer
model.add(Dense(1)) #output layer

model.compile(optimizer='adam',loss='mse') 


# Banyaknya hidden layer itu tidak ada patokannya, akan tetapi semakin rumit permasalahan, semakin banyak hidden layer. 
# `https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw`

# ### Training Model 

# `batch size` adalah jumlah batch datanya (jadi datanya dipecah menjadi beberapa batch), semakin sedikit jumlah batch semakin lama runningnya. Batch berguna untuk data yang berukuran besar. Jumlah batch biasanya 2^n.
# 
# `epochs` adalah iterasi untuk update penimbang dan bias. 1 epoch sama dengan menjalankan network dari input sampai ke output. epoch ke 2 (iterasi ke-2) meng-update penimbang dan bias kemudian menjalankan networknya lagi, dst. Standarnya berapa? kira-kira aja wkwkwk nnti diganti2 aja jumlahnya

# In[23]:


model.fit(x=X_train,
          y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=32,epochs=400)

#kalau ada error, coba di bagian deklarasi variabel X dan y, belakangnya ditambahin/diilangin values(), yg ini misal e X = data.drop('SalePrice', axis=1).values()


# hasil di atas bakal beda2 klo di run ulang yak, karna inisialisasi dan updating nilai penimbang dan bias itu random

# membandingkan loss dengan validation loss

# In[24]:


losses = pd.DataFrame(model.history.history)


# In[25]:


losses


# In[26]:


losses.plot()


# Jika garis oranye semakin lama semakin ke atas, itu berarti overfitting. Jika garis biru yang ke atas, berarti underfitting
# Beberapa cara mengatasinya yaitu mengubuah jumlah epochs, menambah/mengurangi jumlah hidden layer, menambah/mengurangi perceptron/node/neuron.

# ### Model Evaluation 

# In[27]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score


# In[28]:


X_test


# In[29]:


predictions = model.predict(X_test)


# In[30]:


mean_absolute_error(y_test,predictions) #mean absolute error antara y test (nilai asli) dengan y prediction


# In[31]:


np.sqrt(mean_squared_error(y_test,predictions)) #root mean square error antara y test (nilai asli) dengan y prediction


# In[32]:


explained_variance_score(y_test,predictions) #nilai varians yang bisa dijelaskan oleh model


# In[33]:


# Our predictions
plt.scatter(y_test,predictions)

# Perfect predictions
plt.plot(y_test,y_test,'r')


# In[34]:


errors = y_test.values.reshape(6192, 1) - predictions


# In[35]:


sns.distplot(errors)


# ### Predicting

# In[36]:


single_house = df.drop(['SalePrice','Latitude', 'Longitude'],axis=1).iloc[0]
#iloc[0] untuk mengambil data pada baris pertama


# In[37]:


single_house


# In[38]:


single_house = single_house.values.reshape(-1, 6) 


# Kita reshape menjadi bentuk array
# 
# -1 berarti semua variabel ikut
# 
# 6 mksdnya banyak variabel

# In[39]:


single_house


# In[40]:


single_house = scaler.transform(single_house) #minmaxscaler


# In[41]:


model.predict(single_house)


# In[42]:


df.head(1)


# Harga aslinya 4.526, harga yg dipredict 4.3592577
# mayan lah ya

#  
