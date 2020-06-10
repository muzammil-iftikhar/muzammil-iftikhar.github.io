---
title:  "Project: Predicting Amount spent"
date: 2020-06-05
categories: [Project]
tags: [machine learning,data science,python]
excerpt: "Predicting yearly amount spent by the customers of an ecommerce store. We will be using pipelines, cross_validation_scores and LinearRegression model"
author_profile: true
mathjax: true
---

For this project, we are going to use this [dataset](https://www.kaggle.com/iyadavvaibhav/ecommerce-customer-device-usage) from Kaggle. This data is of ecommerce customers and their usage on app vs website. Our mission is to comeup with a model that will predict the 'Yearly Amount Spent' by the customers based on the given features.

Highlights:
* Exploratory data analysis
* Creating pipeline
* Using LinearRegression model
* Using cross-validation scores to measure model performance

```python
#Imports
import numpy as np
import pandas as pd
```

```python
#Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
%matplotlib inline
sns.set()
```

```python
#import data
ecommerce = pd.read_csv('ecommerce')
```

```python
#Glimpse of data
ecommerce.head(2)
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
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Email</th>
      <th>Address</th>
      <th>Avatar</th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mstephenson@fernandez.com</td>
      <td>835 Frank Tunnel\nWrightmouth, MI 82180-9605</td>
      <td>Violet</td>
      <td>34.497268</td>
      <td>12.655651</td>
      <td>39.577668</td>
      <td>4.082621</td>
      <td>587.951054</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hduke@hotmail.com</td>
      <td>4547 Archer Common\nDiazchester, CA 06566-8576</td>
      <td>DarkGreen</td>
      <td>31.926272</td>
      <td>11.109461</td>
      <td>37.268959</td>
      <td>2.664034</td>
      <td>392.204933</td>
    </tr>
  </tbody>
</table>
</div>

```python
#Data info
ecommerce.info()

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 8 columns):
    Email                   500 non-null object
    Address                 500 non-null object
    Avatar                  500 non-null object
    Avg. Session Length     500 non-null float64
    Time on App             500 non-null float64
    Time on Website         500 non-null float64
    Length of Membership    500 non-null float64
    Yearly Amount Spent     500 non-null float64
    dtypes: float64(5), object(3)
    memory usage: 31.3+ KB
```

```python
#describe numerical features
ecommerce.describe()
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
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>33.053194</td>
      <td>12.052488</td>
      <td>37.060445</td>
      <td>3.533462</td>
      <td>499.314038</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.992563</td>
      <td>0.994216</td>
      <td>1.010489</td>
      <td>0.999278</td>
      <td>79.314782</td>
    </tr>
    <tr>
      <th>min</th>
      <td>29.532429</td>
      <td>8.508152</td>
      <td>33.913847</td>
      <td>0.269901</td>
      <td>256.670582</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.341822</td>
      <td>11.388153</td>
      <td>36.349257</td>
      <td>2.930450</td>
      <td>445.038277</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.082008</td>
      <td>11.983231</td>
      <td>37.069367</td>
      <td>3.533975</td>
      <td>498.887875</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>33.711985</td>
      <td>12.753850</td>
      <td>37.716432</td>
      <td>4.126502</td>
      <td>549.313828</td>
    </tr>
    <tr>
      <th>max</th>
      <td>36.139662</td>
      <td>15.126994</td>
      <td>40.005182</td>
      <td>6.922689</td>
      <td>765.518462</td>
    </tr>
  </tbody>
</table>
</div>

Observation:
* Our features seem to be normally distributed as mean is very close to median values
* The feature data is spread very close to the mean as standard deviation is very low

```python
#Drop the non-numerical columns
ecommerce.drop(['Address','Avatar','Email'],axis=1,inplace=True)
```

```python
#Check for null values
[col for col in ecommerce.columns if ecommerce[col].isnull().any()]

    []
```

Seems like there are no null values. Makes our life easier :)

### Exploratory data analysis

Let's explore our data a bit

```python
sns.distplot(ecommerce['Avg. Session Length'],label='Avg. session length')
sns.distplot(ecommerce['Time on App'],label='Time on app')
sns.distplot(ecommerce['Time on Website'],label='Time on Website')
sns.distplot(ecommerce['Length of Membership'],label='Length of membership')
plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    <matplotlib.legend.Legend at 0x167f8cb3c50>
```

![png](/projects/Ecommerce_device_usage/images/Ecommerce_device_usage_16_1.png)

Distribution plot above proves our first observation that features are normally distributed.

```python
plt.figure(figsize=(10,5))
sns.boxenplot(data=ecommerce.drop('Yearly Amount Spent',axis=1))
plt.tight_layout()
```

![png](/projects/Ecommerce_device_usage/images/Ecommerce_device_usage_18_0.png)

Above Boxplot proves our second observation that the features data spread is very close to the mean.

```python
#Pair plot
sns.pairplot(ecommerce)

    <seaborn.axisgrid.PairGrid at 0x167fb0f9a58>
```

![png](/projects/Ecommerce_device_usage/images/Ecommerce_device_usage_20_1.png)

There seems to be a very strong relationship b/w length of membership and our target label.

```python
sns.heatmap(ecommerce.corr(),cmap='magma_r',annot=True)

    <matplotlib.axes._subplots.AxesSubplot at 0x167fbcf1748>
```

![png](/projects/Ecommerce_device_usage/images/Ecommerce_device_usage_22_1.png)

Heatmap proves that relationship by showing us the pearson's r value of 0.81. There also seems to be some relation between `Time on App` and our target variable.

```python
sns.jointplot(x='Length of Membership',y='Time on App',data=ecommerce,kind='hex')

    <seaborn.axisgrid.JointGrid at 0x167fc21a940>
```

![png](/projects/Ecommerce_device_usage/images/Ecommerce_device_usage_24_1.png)

```python
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=ecommerce)

    <seaborn.axisgrid.FacetGrid at 0x167fc7c0668>
```

![png](/projects/Ecommerce_device_usage/images/Ecommerce_device_usage_25_1.png)

### Defining pipeline

```python
X = ecommerce.drop('Yearly Amount Spent',axis=1)
y = ecommerce['Yearly Amount Spent']
```

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
```
```python
my_pipeline = make_pipeline(LinearRegression())
```

### Using cross-validation scores

```python
from sklearn.model_selection import cross_val_score
```

```python
cv_scores = -1 * cross_val_score(my_pipeline,X,y,cv=5,scoring='neg_mean_absolute_error')
```

```python
cv_scores.mean()

    7.944690345653413
```
