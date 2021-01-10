#!/usr/bin/env python
# coding: utf-8

# # Deep Learning for Image Classification

# Data yang digunakan adalah datasets gambar tentang fashion dari library keras

#     Label    Description
#     0        T-shirt/top
#     1        Trouser
#     2        Pullover
#     3        Dress
#     4        Coat
#     5        Sandal
#     6        Shirt
#     7        Sneaker
#     8        Bag
#     9        Ankle boot
# 
# Ada 60k training data dan 10k testing data

# In[1]:


from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# ## Visualisasi Data

# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


x_train[0]


# In[4]:


plt.imshow(x_train[0])


# Setiap gambar terdiri dari 28 x 28 array yang bernilai antara 0-255, yang merepresentasikan warna dalam rgb (dalam kasus ini hanyalah warna hitam dan putih, untuk gambar di atas warna kuning hijau ungu itu bawaan dari python).

# In[5]:


y_train[0]


# Label 9 adalah sepatu, bisa di lihat pada keterangan di paling atas

# ## Preprocessing the Data

# Normalisasi data (jadi nilai dari tiap array hanya 0-1 saja).

# In[6]:


x_train.max()


# In[7]:


x_train = x_train/255
x_test = x_test/255


# In[8]:


x_train.shape


# In[9]:


x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#60k untuk ukuran data
#28,28 untuk ukuran pixel
#1 untuk hitam putih, isi 3 jika RGB


# One Hot Encoding

# In[10]:


from tensorflow.keras.utils import to_categorical


# In[11]:


y_train


# In[12]:


y_cat_train = to_categorical(y_train)


# In[13]:


y_cat_test = to_categorical(y_test)


# # Building the model

# * 2D Convolutional Layer, filters=32 and kernel_size=(4,4)
# * Pooling Layer di mana pool_size = (2,2)
# 
# * Flatten Layer
# * Dense Layer (128 Neurons, ini terserah boleh diubah2, nilainya 2^n yaa sabeb), RELU activation
# 
# * Final Dense Layer of 10 Neurons with a softmax activation

# In[14]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# In[15]:


model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES 
model.add(Flatten())

# 128 NEURONS DI DENSE HIDDEN LAYER (BOLEH DIUBAH)
model.add(Dense(128, activation='relu'))

# TERAKHIR LAYER KLASIFIKASI, ADA 10 KATEGORI
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[16]:


model.summary()


# In[17]:


model.fit(x_train,y_cat_train,epochs=10)


# # Evaluasi Model

# In[18]:


from sklearn.metrics import classification_report


# In[19]:


predictions = model.predict_classes(x_test)


# In[20]:


y_cat_test.shape


# In[21]:


y_cat_test[0] 


# In[22]:


predictions[0] #kite pen ngepredict gambar pertama itu gambar apa


# In[23]:


y_test #hasilnya sama yak, labelnya sama-sama 9, yaitu ankle boot


# In[24]:


print(classification_report(y_test,predictions))

