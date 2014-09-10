# Practical Machine Learning
Weijia Chen  
10 September 2014  

## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. This report investigated and analysed a dataset which was collected from accelermeters on the belt, forearm, arm, and dumbell of  six participants. Thoes volunteers were asked to perform barbell lifts correctly and incorrectly in 5 different ways. Then, a classification prediction model was built by using the training data to classify participants' ways in lifting the barbell. Finally, the prediction model was applied to the testing dataset to test the prediction performance.  

##Getting and cleaning the data for prediction
Set working directory, load library


```r
setwd("~/Practical-Machine-Learning")
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(tree)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(corrplot)
```
Load both train and test data, replace blank and "#DIV/0!" to NA


```r
traindata <- read.csv("pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
testdata <- read.csv("pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))
```
Identify all columns with NA in traindata


```r
trainna <- apply(traindata, 2, function(x) sum(is.na(x)))
trainnona <- names(trainna)[trainna==0] #identify all variables without na
traindata <- traindata[, trainnona] #subset traindata with all variables without na
traindata <- traindata[, -c(1:7)]   #remove first seven column
```
Identify all columns with NA in testdata


```r
testna <- apply(testdata, 2, function(x) sum(is.na(x)))
testnona <- names(testna)[testna==0]
testdata <- testdata[, testnona]
testdata <- testdata[, -c(1:7)]
```

##Prediction methodology

Divide the traindata in order to create a cross validation set. 70 per cent traindata will be used to subtrain data, and 40 per cent to cross validation set. The aim of prediction model is to predict class types. Therefore, variable classe will be used.


```r
set.seed(6897)
inTrain <- createDataPartition(y=traindata$classe, p=.60, list=FALSE)
trainingdata <- traindata[inTrain,]
training.cv <- traindata[-inTrain,]
```

###Random forests
First, random forests will be used to train the trainingdata, then, trained model will be tested by validation set training.cv 

```r
fit <- randomForest(classe ~ ., data = trainingdata, method = "class")
predictor <- predict(fit, training.cv)
confusionMatrix(predictor, training.cv$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2227    6    0    0    0
##          B    5 1508   17    0    0
##          C    0    4 1350   16    0
##          D    0    0    1 1269    7
##          E    0    0    0    1 1435
## 
## Overall Statistics
##                                         
##                Accuracy : 0.993         
##                  95% CI : (0.991, 0.994)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.991         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.993    0.987    0.987    0.995
## Specificity             0.999    0.997    0.997    0.999    1.000
## Pos Pred Value          0.997    0.986    0.985    0.994    0.999
## Neg Pred Value          0.999    0.998    0.997    0.997    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.172    0.162    0.183
## Detection Prevalence    0.285    0.195    0.175    0.163    0.183
## Balanced Accuracy       0.998    0.995    0.992    0.993    0.997
```

```r
sum(predictor == training.cv$classe) / length(predictor)
```

```
## [1] 0.9927
```
Based on the results, it can be clearly see that with the **_99%_** validation error, the performance of prediction model is good.

##Output
Apply the prediction model with testdata set.

```r
results <- predict(fit, newdata = testdata)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_", i, ".txt")
    write.table(x[i], file = filename, quote = FALSE, row.names = FALSE,col.names = FALSE)
  }
}

pml_write_files(results)
```

##Conclusion
In this report, randomForest was used to creat a prediction model. The performance of prediction model is good in classifying the twenty classification that correctly matched the test set values.






