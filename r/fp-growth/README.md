# Fp-Growth <img src="https://img.shields.io/badge/r-%23276DC3.svg?&style=for-the-badge&logo=r&logoColor=white"/> 


Load Library
============

library yang dibutuhkan, yaitu **rCBA**. Jika belum terinstall, silahkan
install terlebih dahulu dengan perintah `install.packages("rCBA")`.

``` r
library(rCBA)
library(knitr)
```

Load Data
=========

``` r
data <- read.csv("beli_komputer.csv", header=T, sep = ";")
kable(head((data)))
```

|   id| age         | income | student | credit\_rating | Buy\_Computer |
|----:|:------------|:-------|:--------|:---------------|:--------------|
|    1| youth       | high   | no      | fair           | no            |
|    2| youth       | high   | no      | excellent      | no            |
|    3| middle\_age | high   | no      | fair           | yes           |
|    4| senior      | medium | no      | fair           | yes           |
|    5| senior      | low    | yes     | fair           | yes           |
|    6| senior      | low    | yes     | excellent      | no            |

Konversi Data
=============

Ubah tipe variabel menjadi tipe faktor

``` r
data <- data[-1]
for(i in names(data)){
  data[,i]= as.factor(data[,i])
}
str(data)
```

    ## 'data.frame':    14 obs. of  5 variables:
    ##  $ age          : Factor w/ 3 levels "middle_age","senior",..: 3 3 1 2 2 2 1 3 3 2 ...
    ##  $ income       : Factor w/ 3 levels "high","low","medium": 1 1 1 3 2 2 2 3 2 3 ...
    ##  $ student      : Factor w/ 2 levels "no","yes": 1 1 1 1 2 2 2 1 2 2 ...
    ##  $ credit_rating: Factor w/ 2 levels "excellent","fair": 2 1 2 2 2 1 1 2 2 2 ...
    ##  $ Buy_Computer : Factor w/ 2 levels "no","yes": 1 1 2 2 2 1 2 1 2 2 ...

Buat Model
==========

``` r
data <- data.frame(data, check.names = FALSE)
trans <- as(data, "transactions")
rules <- fpgrowth(trans, support = 0.2, confidence = 0.7, maxLength = 2, 
                      consequent = "Buy_Computer", parallel = FALSE)
inspect(rules)
```

    ##     lhs                     rhs                support   confidence lift    
    ## [1] {age=middle_age}     => {Buy_Computer=yes} 0.2857143 1.0000000  1.555556
    ## [2] {income=low}         => {Buy_Computer=yes} 0.2142857 0.7500000  1.166667
    ## [3] {student=yes}        => {Buy_Computer=yes} 0.4285714 0.8571429  1.333333
    ## [4] {credit_rating=fair} => {Buy_Computer=yes} 0.4285714 0.7500000  1.166667

``` r
prediksi <- classification(data, rules)
```

    ## 2021-01-10 20:39:22 rCBA: initialized

    ## 2021-01-10 20:39:22 rCBA: data 14x5

    ##   took: 0  s

    ## 2021-01-10 20:39:22 rCBA: rules 4

    ##   took: 0  s

    ## 2021-01-10 20:39:22 rCBA: classification completed

    ##   took: 0.02  s

``` r
table(prediksi)
```

    ## prediksi
    ## yes 
    ##  12

Penerapan Pada Data Iris
========================

Buat Model
----------

``` r
data("iris")
train <- sapply(iris,as.factor)
train <- data.frame(train, check.names=FALSE)
txns <- as(train,"transactions")
rules <- fpgrowth(txns, support=0.05, confidence=0.05, maxLength=2, consequent="Species",
           parallel=FALSE)
inspect(rules)
```

    ##      lhs                   rhs                  support    confidence lift     
    ## [1]  {Petal.Width=2.3}  => {Species=virginica}  0.05333333 1.0000000  3.0000000
    ## [2]  {Sepal.Length=5.1} => {Species=setosa}     0.05333333 0.8888889  2.6666667
    ## [3]  {Sepal.Length=5}   => {Species=setosa}     0.05333333 0.8000000  2.4000000
    ## [4]  {Sepal.Width=3.4}  => {Species=setosa}     0.06000000 0.7500000  2.2500000
    ## [5]  {Petal.Width=1.8}  => {Species=virginica}  0.07333333 0.9166667  2.7500000
    ## [6]  {Petal.Width=1.5}  => {Species=versicolor} 0.06666667 0.8333333  2.5000000
    ## [7]  {Petal.Length=1.4} => {Species=setosa}     0.08666667 1.0000000  3.0000000
    ## [8]  {Petal.Length=1.5} => {Species=setosa}     0.08666667 1.0000000  3.0000000
    ## [9]  {Petal.Width=1.3}  => {Species=versicolor} 0.08666667 1.0000000  3.0000000
    ## [10] {Sepal.Width=2.8}  => {Species=virginica}  0.05333333 0.5714286  1.7142857
    ## [11] {Sepal.Width=3}    => {Species=versicolor} 0.05333333 0.3076923  0.9230769
    ## [12] {Sepal.Width=3}    => {Species=virginica}  0.08000000 0.4615385  1.3846154
    ## [13] {Petal.Width=0.2}  => {Species=setosa}     0.19333333 1.0000000  3.0000000
    ## [14] {}                 => {Species=setosa}     0.33333333 0.3333333  1.0000000
    ## [15] {}                 => {Species=virginica}  0.33333333 0.3333333  1.0000000
    ## [16] {}                 => {Species=versicolor} 0.33333333 0.3333333  1.0000000

Prediksi
--------

``` r
predictions <- classification(train,rules)
table(predictions)
```

    ## predictions
    ##     setosa versicolor  virginica 
    ##         86         25         39

Akurasi
-------

``` r
sum(as.character(train$Species)==as.character(predictions),na.rm=TRUE)/length(predictions)
```

    ## [1] 0.6866667

Prunning
--------

``` r
prunedRules <- pruning(train, rules, method="m2cba", parallel=FALSE)
inspect(prunedRules)
```

    ##      lhs                   rhs                  support    confidence lift    
    ## [1]  {Petal.Width=0.2}  => {Species=setosa}     0.19333333 1.0000000  3.000000
    ## [2]  {Petal.Length=1.4} => {Species=setosa}     0.08666667 1.0000000  3.000000
    ## [3]  {Petal.Length=1.5} => {Species=setosa}     0.08666667 1.0000000  3.000000
    ## [4]  {Petal.Width=1.3}  => {Species=versicolor} 0.08666667 1.0000000  3.000000
    ## [5]  {Petal.Width=2.3}  => {Species=virginica}  0.05333333 1.0000000  3.000000
    ## [6]  {Petal.Width=1.8}  => {Species=virginica}  0.07333333 0.9166667  2.750000
    ## [7]  {Sepal.Length=5.1} => {Species=setosa}     0.05333333 0.8888889  2.666667
    ## [8]  {Petal.Width=1.5}  => {Species=versicolor} 0.06666667 0.8333333  2.500000
    ## [9]  {Sepal.Length=5}   => {Species=setosa}     0.05333333 0.8000000  2.400000
    ## [10] {}                 => {Species=virginica}  0.33333333 0.3333333  1.000000

### Prediksi Setelah Prunning

``` r
predictions <- classification(train, prunedRules)
table(predictions)
```

    ## predictions
    ##     setosa versicolor  virginica 
    ##         48         25         77

### Akurasi Setalah Prunning

``` r
sum(as.character(train$Species)==as.character(predictions),na.rm=TRUE)/length(predictions)
```

    ## [1] 0.7733333

Terlihat Bahwa akurasinya meningkat
