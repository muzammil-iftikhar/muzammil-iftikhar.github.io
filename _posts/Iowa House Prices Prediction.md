---
title:  "Project: Iowa House Prices Prediction"
date: 2020-05-15
categories: [Tutorial]
tags: [machine learning,data science,python]
excerpt: "Predict the prices of Iowa houses using DecisionTreeRegressor"
author_profile: true
---

In this project, we will be predicting the house prices of Iowa houses. Let's get started

We will be using the setps that were referenced here [SML:Supervised Machine Learning workflow](https://muzammil-iftikhar.github.io/reading/Machine-Learning-flow/)

The data set that we are going to use is the Iowa house prices dataset from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

### Define Problem

Since we are to predict the house prices, it is going to be a regression problem.  
Read about [Regression vs Classification]() problems here.

### Acquire Data

Go ahead and download the dataset from the Kaggle link above

### Import Data


```python
%%HTML
<style type="text/css">
table.dataframe td, table.dataframe th {
    border: 1px  black solid !important;
  color: black !important;
}
</style>
<!--This is just to make dataframes inside jupyter with borders-->
```


<style type="text/css">
table.dataframe td, table.dataframe th {
    border: 1px  black solid !important;
  color: black !important;
}
</style>
<!--This is just to make dataframes inside jupyter with borders-->




```python
#Importing necessary libraries
import numpy as np
import pandas as pd
```


```python
#Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
%matplotlib inline
sns.set()
```


```python
iowa = pd.read_csv("C:/Users/muzam/Desktop/muzammil-iftikhar.github.io/Projects/Iowa house prices/train.csv")
```


```python
iowa.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



### Exploratory Data Analysis

Lets explore our data a bit and see what do we have in hand


```python
len(iowa.columns)
```




    81



There are 81 columns, for the sake of this project and for understanding purposes, we will use only following features:
LotFrontage, LotArea, Utilities, BldgType, HouseStyle, YearBuilt, 1stFlrSF, 2ndFlrSF, BedroomAbvGr, YrSold, SaleType, SalePrice


```python
features = ['LotFrontage','LotArea','Utilities','BldgType','HouseStyle','YearBuilt','1stFlrSF','2ndFlrSF','BedroomAbvGr','YrSold','SaleType']
target = 'SalePrice'
#Feature Dataframe
iowa_feat = iowa[features]
#Target Dataframe
iowa_tar = iowa[target]
iowa = pd.concat([iowa_feat,iowa_tar],axis=1)
```


```python
iowa_feat.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Utilities</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>YearBuilt</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>BedroomAbvGr</th>
      <th>YrSold</th>
      <th>SaleType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65.0</td>
      <td>8450</td>
      <td>AllPub</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>2003</td>
      <td>856</td>
      <td>854</td>
      <td>3</td>
      <td>2008</td>
      <td>WD</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80.0</td>
      <td>9600</td>
      <td>AllPub</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>1976</td>
      <td>1262</td>
      <td>0</td>
      <td>3</td>
      <td>2007</td>
      <td>WD</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68.0</td>
      <td>11250</td>
      <td>AllPub</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>2001</td>
      <td>920</td>
      <td>866</td>
      <td>3</td>
      <td>2008</td>
      <td>WD</td>
    </tr>
  </tbody>
</table>
</div>




```python
#get the idea of number of rows and columns and type of data in each
iowa_feat.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 11 columns):
    LotFrontage     1201 non-null float64
    LotArea         1460 non-null int64
    Utilities       1460 non-null object
    BldgType        1460 non-null object
    HouseStyle      1460 non-null object
    YearBuilt       1460 non-null int64
    1stFlrSF        1460 non-null int64
    2ndFlrSF        1460 non-null int64
    BedroomAbvGr    1460 non-null int64
    YrSold          1460 non-null int64
    SaleType        1460 non-null object
    dtypes: float64(1), int64(6), object(4)
    memory usage: 125.5+ KB
    


```python
iowa_feat.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>YearBuilt</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>BedroomAbvGr</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>1971.267808</td>
      <td>1162.626712</td>
      <td>346.992466</td>
      <td>2.866438</td>
      <td>2007.815753</td>
    </tr>
    <tr>
      <th>std</th>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>30.202904</td>
      <td>386.587738</td>
      <td>436.528436</td>
      <td>0.815778</td>
      <td>1.328095</td>
    </tr>
    <tr>
      <th>min</th>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1872.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2006.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>1954.000000</td>
      <td>882.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2007.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>1973.000000</td>
      <td>1087.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>2008.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>2000.000000</td>
      <td>1391.250000</td>
      <td>728.000000</td>
      <td>3.000000</td>
      <td>2009.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>2010.000000</td>
      <td>4692.000000</td>
      <td>2065.000000</td>
      <td>8.000000</td>
      <td>2010.000000</td>
    </tr>
  </tbody>
</table>
</div>



Observations:
- LotFrontage
    - Has only 1201 values, which means that there are missing values. We will handle that in the next stage
    - Has mean of 70 ft.
    - Has standard deviation of around 24 ft.
    - Max value seems like an outlier as 75% of the data is within 80 ft.
- LotArea
    - Seems to be highly right skewed distribution
    - Max value is definitely an outlier
- YearBuilt
    - Max is 2010 which means either no house was sold after 2010 or the data was only collected upto 2010
- 1stFlrSF
    - Seems to be evenly distributed
- 2ndFlrSF
    - Upto 50% of the data here has 0 value which means that around 50% of the houses are single storey
- BedroomAbvGr
    - Min is 0 rooms
- YrSold
    - Houses were sold from 2006 to 2010

**Observing Categorical Variables**


```python
iowa_feat.describe(include=['O'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Utilities</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>SaleType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
      <td>1460</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>5</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>top</th>
      <td>AllPub</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>WD</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1459</td>
      <td>1220</td>
      <td>726</td>
      <td>1267</td>
    </tr>
  </tbody>
</table>
</div>



- Utilities:
    -Almost all of the houses belong to single category. I don't see it affecting our house prices. We may drop it in next stage
- BldgType:
    -Almost 83% of data points fall in a single category here as well. We might drop this column as well
- HouseStyle:
    -There seems to be somewhat distribution among multiple categories here. We will include this in our predictions
- SaleType:
    -Almost 80% of data points fall in a single category. We will drop this column as well

**Visualizing Data**

Here we will visualize our categorical and numerical variables and confirm some of the descriptive observations that we made above


```python
sns.countplot(data=iowa,x='Utilities')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249df9fff60>




![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_28_1.png)


This confirms our above observation that almost all of the data points belong to single category of Utilities


```python
sns.countplot(data=iowa,x='BldgType')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249df7dff98>




![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_30_1.png)


This confirms our observation that 'BldgType' doesn't seem to much impact the target price as well since about 80% of data is in single category. We will drop this also.


```python
sns.countplot(data=iowa,x='HouseStyle')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249dc69dba8>




![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_32_1.png)


This confirms our observation about 'HouseStyle' that there is some distribution here. We will include it in our predictions


```python
sns.countplot(data=iowa,x='SaleType')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249dc61f390>




![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_34_1.png)


This confirms our observation that almost all of the data points fall under a single category in 'SaleType'


```python
sns.boxplot(data=iowa,x='LotFrontage',palette='rainbow')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249df89ff98>




![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_36_1.png)


We were right about our observation that 313 is the outlier. Around 50% of data falls between 50 and 100.


```python
plt.figure(figsize=(15,5))
sns.boxplot(data=iowa,x='LotArea',palette='rainbow')
plt.tight_layout()
```


![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_38_0.png)



```python
plt.figure(figsize=(15,5))
sns.distplot(a=iowa['LotArea'],bins=100,kde=False,rug=True)
plt.tight_layout()
```


![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_39_0.png)


Above boxplot and hist plot of 'LotArea' confirms our observations that the distribution is highly right skewed. Max value is also an outlier


```python
iowa['1stFlrSF'].plot(kind='hist',bins=30)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249e513c0f0>




![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_41_1.png)


This confirms our observation about '1stFlrSF'. The distribution is good. We will include this feature in our predictions


```python
iowa['2ndFlrSF'].plot(kind='hist',bins=30)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249e4c23c88>




![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_43_1.png)


This confirms our observation about '2ndFlrSF'. Most of the data points are with 0 value


```python
iowa['BedroomAbvGr'].value_counts().sort_index().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249e50f6e48>




![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_45_1.png)



```python
iowa['YrSold'].value_counts().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249e5382240>




![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_46_1.png)



```python
sns.pairplot(data=iowa)
```




    <seaborn.axisgrid.PairGrid at 0x249c7ebcbe0>




![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_47_1.png)



```python
sns.heatmap(iowa.corr(),cmap='magma_r',annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249eb4f1438>




![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_48_1.png)


YrSold has almost 0 pearson's r value in correlation to SalePrice. We will drop this column also.  
LotFrontage has some relation to SalePrice, we will keep it and fill it up

### Data Cleaning, Data Completing, Feature Engineering

Lets clean the data by dropping the columns we have decided to drop in our above analysis.


```python
iowa_feat.drop(['Utilities','BldgType','SaleType','YrSold'],axis=1,inplace=True)
```

Lets complete the data now by filling in the null values


```python
#check for null values in our features
plt.figure(figsize=(10,5))
sns.heatmap(iowa_feat.isnull(),cbar=False,yticklabels=False,cmap='viridis')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249eca01c18>




![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_54_1.png)


So, only LotFrontage has null values. Lets complete them by fillig them with the average value. Do remember that there are alot of ways to fill in the values, but for now, i will just fill them up with the mean value of the column


```python
iowa_feat['LotFrontage'].fillna(iowa_feat['LotFrontage'].mean(),inplace=True)
```


```python
#check for null values again
plt.figure(figsize=(10,5))
sns.heatmap(iowa_feat.isnull(),cbar=False,yticklabels=False,cmap='viridis')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x249ef63db00>




![png](Iowa%20House%20Prices%20Prediction_files/Iowa%20House%20Prices%20Prediction_57_1.png)


Great, now that we dont have any null values, let's proceed

Now, we need to convert one of the features to numerical values as part of feature engineering


```python
iowa_feat = pd.get_dummies(iowa_feat,drop_first=True)
```

### Get Model

We will be using the DecisionTreeRegressor model in this project. There are alot of other models out there that may generate better results and we will get our hands dirty with them in future.


```python
from sklearn.tree import DecisionTreeRegressor
```


```python
dtr = DecisionTreeRegressor()
```

### Train/Fit Model

Now, for this step we need to first split our dataset into training and testing. Note that when you download the dataset from Kaggle, they have done it for you and you won't have to do it yourself.  
But i want to show you how it's done. Remember, you always train your model with the training dataset and you test it with the test data. You will never validate/test your model with the training dataset


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(iowa_feat, iowa_tar, test_size=0.3, random_state=101)
```


```python
#Lets fit our model with the training dataset
dtr.fit(X_train,y_train)
```




    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
               max_leaf_nodes=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               presort=False, random_state=None, splitter='best')



### Test Model


```python
predictions = dtr.predict(X_test)
#we always test the model with testing dataset
```

### Validate Model

Lets check the performance of our model


```python
from sklearn.metrics import mean_absolute_error
```


```python
mean_absolute_error(y_test,predictions)
```




    30866.23515981735



Which means that our predictions are on an average around 31k USD from the actual values y_test

Now we can make it better by using the concept of bias-variance tradeoff. If you go and have a look at the [steps](https://muzammil-iftikhar.github.io/reading/Machine-Learning-flow/), you will see that once we validate our model, we either go and get a new model to we retrain our model with different parameters to get better predictions. In this project, we will retrain our model


```python
max_depth = [2,5,10,15,50,100,500]
mae = []
for n in max_depth:
    dtr = DecisionTreeRegressor(max_depth=n)
    dtr.fit(X_train,y_train)
    predictions = dtr.predict(X_test)
    mae.append(mean_absolute_error(y_test,predictions))
```


```python
mae
```




    [40620.180940294726,
     29823.564573629337,
     29233.503839113608,
     31147.21811601365,
     30481.737442922375,
     30627.96803652968,
     31053.470319634704]



So we get the best results when max_depth = 10
