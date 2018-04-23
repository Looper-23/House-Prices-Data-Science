# House-Prices-Data-Science

I tried a lot of techniques in order to solve the problem of finding a model which can be applied to testing file. To start with, I had to decide how I was going to deal with the missing data. If I had taken the approach of deleting the rows that had missing data, I was sure I wasn’t going to get the best fit. Especially looking at other kernels on Kaggle, most went deep into how deleting the data skews the answer a lot. So, I had to take the “Imputing” approach from “sklearn” library. For that I used the notebook from Victor Henrique notebook which showed how you can impute all the missing data with the means of those columns. My next step was to look at the “Heatmap” to see which variables are most related to the price in the training set. Using those I drew some plots and regressions using seaborn and matplotlib to visualize how I can normalize the data. And in the end I applied the Sklearn linear models, repressors and ensemble to get the model and apply it to my test file. 


This model put at the 1000th position in the Kaggle competion. 
