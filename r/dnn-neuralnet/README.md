# Deep Neural Network - Neuralnet <img src="https://img.shields.io/badge/r-%23276DC3.svg?&style=for-the-badge&logo=r&logoColor=white"/> 


Load Library
============

Tiga library yang dibutuhkan, yaitu **neuralnet dan caret**. Jika belum
terinstall, silahkan install terlebih dahulu dengan perintah
`install.packages("nama-package")`.

Library **neuralnet** akan digunakan untuk membuat model neural network.
Library **caret** digunakan untuk membuat confusion matriks dan melihar
akurasi model.

``` r
library(caret)
library(neuralnet)
library(knitr)
```

Baca Data
=========

``` r
ipeh <- read.csv("data_Ipeh2.csv", header=T)
kable(head(ipeh))
```

|  admit|  gre|   gpa|  rank|
|------:|----:|-----:|-----:|
|      0|  380|  3.61|     3|
|      1|  660|  3.67|     3|
|      1|  800|  4.00|     1|
|      1|  640|  3.19|     4|
|      0|  520|  2.93|     4|
|      1|  760|  3.00|     2|

Preprocessing
=============

### Liat Struktur Data

``` r
str(ipeh)
```

    ## 'data.frame':    400 obs. of  4 variables:
    ##  $ admit: int  0 1 1 1 0 1 1 0 1 0 ...
    ##  $ gre  : int  380 660 800 640 520 760 560 400 540 700 ...
    ##  $ gpa  : num  3.61 3.67 4 3.19 2.93 3 2.98 3.08 3.39 3.92 ...
    ##  $ rank : int  3 3 1 4 4 2 1 2 3 2 ...

Normalisasi
-----------

Menggunakan min-max scaler

``` r
normalisasi <- function(r){
  return((r-min(r))/(max(r)-min(r)))
}

for(i in colnames(ipeh[-1])){
    ipeh[ ,i]=normalisasi(ipeh[ ,i])
}

str(ipeh)
```

    ## 'data.frame':    400 obs. of  4 variables:
    ##  $ admit: int  0 1 1 1 0 1 1 0 1 0 ...
    ##  $ gre  : num  0.276 0.759 1 0.724 0.517 ...
    ##  $ gpa  : num  0.776 0.81 1 0.534 0.385 ...
    ##  $ rank : num  0.667 0.667 0 1 1 ...

Split Data
----------

Memecah data menjadi data training (80% dari data awal) dan data test
(20% dari data awal)

``` r
set.seed(2700)
sampel <- sample(2,nrow(ipeh),replace = T, prob = c(0.8,0.2))
trainingdat <- ipeh[sampel==1, ]
testingdat <- ipeh[sampel==2, ]
print(paste("Jumlah train data :", nrow(trainingdat), paste("|| Jumlah test data :", nrow(testingdat))))
```

    ## [1] "Jumlah train data : 326 || Jumlah test data : 74"

Membuat Model
=============

Deep neural network adalah istilah neural network untuk hidden layer 2
atau lebih. Semakin banyak hidden layer semakin kompleks model tersebut
tetapi tidak menjamin akan semakin baik akurasinya

*Ket : saya tidak tau kenapa untuk ukuran hidden layer yang besar fungsi
ini menghasilkan warning. Sebaiknya dicoba sendiri dengan ukuran hidden
layer yang besar.*

Training Model
--------------

Misal kita ingin menggunakan semua atributnya

``` r
#model dengan 2 hidden layer, masing masing 2 hidden node dan 1 hidden node
set.seed(0207)
modelnn21 <- neuralnet(admit~gre+gpa+rank, data=trainingdat,
                   hidden = c(2,1),
                   err.fct = "ce",
                   linear.output = F)
plot(modelnn21)
```

**err.fct** merupakan loss function, fungsi yang digunakan untuk melihat
seberapa besar error/loss yang dilakukan model dalam memprediksi,
pilihan fungsi berupa sum square error **“sse”**, atau cross entropy
**“ce”**.

**hidden** merupakan banyaknya hidden layer dan hidden node pada hidden
layer yang akan dibuat. defaultnya, hanya terdapat satu hidden layer dan
satu hidden node. jika ingin mengubah banyaknya hidden layer dan hidden
node tiap layer, gunakan list (contoh hidden = c(5,4) , artinya terdapat
dua hidden layer, hidden layer 1 mempunyai 5 hidden node, hidden layer 2
memiliki 4 hidden node).semakin banyak hidden node dan layer, komputasi
yang dilakukan semakin mahal, namun bisa mengurangi error.

garis dan node biru merupakan bias dan penimbangnya.

**set.seed** diperlukan untuk menyimpan nilai penimbang yang random.
jika tidak digunakan, penimbang yang digunakan akan terus berbeda beda
setiap menjalankan perintah **neuralnet**.

fungsi aktivasi default adalah fungsi sigmoid, untuk mengubah fungsi
aktivasi gunakan atribut **act.fct**. fungsi lain yang tersedia adalah
fungsi tangent hyperbolic **“tanh”**.

baca atribut lain lebih lanjut dengan menjalankan **?neuralnet**

Buat Prediksi
-------------

jika output dari model lebih dari 0.5, maka kategorikan sebagai 1
(admitted), dan 0 (non admitted) jika lainnya

``` r
#2 hidden layer, 2 hidden node dan 1 hidden node
prediksi21 <- compute(modelnn21, testingdat[ ,-1])
pred21 <- ifelse(prediksi21$net.result>0.5, 1, 0)

kable(table(pred21))
```

| pred21 |  Freq|
|:-------|-----:|
| 0      |    63|
| 1      |    11|

Evaluasi Model
--------------

Hidden layer, 2 hidden node dan 1 hidden node

``` r
confusionMatrix(table(pred21, testingdat$admit))
```

    ## Confusion Matrix and Statistics
    ## 
    ##       
    ## pred21  0  1
    ##      0 49 14
    ##      1  5  6
    ##                                           
    ##                Accuracy : 0.7432          
    ##                  95% CI : (0.6284, 0.8378)
    ##     No Information Rate : 0.7297          
    ##     P-Value [Acc > NIR] : 0.45588         
    ##                                           
    ##                   Kappa : 0.2416          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.06646         
    ##                                           
    ##             Sensitivity : 0.9074          
    ##             Specificity : 0.3000          
    ##          Pos Pred Value : 0.7778          
    ##          Neg Pred Value : 0.5455          
    ##              Prevalence : 0.7297          
    ##          Detection Rate : 0.6622          
    ##    Detection Prevalence : 0.8514          
    ##       Balanced Accuracy : 0.6037          
    ##                                           
    ##        'Positive' Class : 0               
    ##
