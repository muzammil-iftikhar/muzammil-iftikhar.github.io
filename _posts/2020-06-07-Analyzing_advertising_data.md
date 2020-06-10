---
title:  "Project: Predicting Ad clicks"
date: 2020-06-07
categories: [Project]
tags: [machine learning,data science,python]
excerpt: "Predicting if a customer is going to click on Ad or not. We will be using pipelines, cross_validation_scores, LogisticRegression and XGBoost"
author_profile: true
mathjax: true
---

In this project, we are going to use this [dataset](https://www.kaggle.com/tbyrnes/advertising) from Kaggle. We will be modeling whether a user is going to click on an Ad or not based on the given features.

Highlights:
* Exploratory data analysis
* Defining pipeline
* Logistic regression model
* Cross validation scores
* XGBoost Classifier model
* XGBoost Randomforest model

```python
#Necessary imports
import numpy as np
import pandas as pd
```

```python
#Visualization libraries
import matplotlib.pyplot as plt
import seaborn as  sns
```

```python
%matplotlib inline
sns.set()
```

```python
#Read dataset
advert = pd.read_csv('advertisiment.csv')
```

```python
#glimpse of data
advert.head(3)
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
      <th>Daily Time Spent on Site</th>
      <th>Age</th>
      <th>Area Income</th>
      <th>Daily Internet Usage</th>
      <th>Ad Topic Line</th>
      <th>City</th>
      <th>Male</th>
      <th>Country</th>
      <th>Timestamp</th>
      <th>Clicked on Ad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>68.95</td>
      <td>35</td>
      <td>61833.90</td>
      <td>256.09</td>
      <td>Cloned 5thgeneration orchestration</td>
      <td>Wrightburgh</td>
      <td>0</td>
      <td>Tunisia</td>
      <td>2016-03-27 00:53:11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80.23</td>
      <td>31</td>
      <td>68441.85</td>
      <td>193.77</td>
      <td>Monitored national standardization</td>
      <td>West Jodi</td>
      <td>1</td>
      <td>Nauru</td>
      <td>2016-04-04 01:39:02</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69.47</td>
      <td>26</td>
      <td>59785.94</td>
      <td>236.50</td>
      <td>Organic bottom-line service-desk</td>
      <td>Davidton</td>
      <td>0</td>
      <td>San Marino</td>
      <td>2016-03-13 20:35:42</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

```python
#Information about data
advert.info()

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 10 columns):
    Daily Time Spent on Site    1000 non-null float64
    Age                         1000 non-null int64
    Area Income                 1000 non-null float64
    Daily Internet Usage        1000 non-null float64
    Ad Topic Line               1000 non-null object
    City                        1000 non-null object
    Male                        1000 non-null int64
    Country                     1000 non-null object
    Timestamp                   1000 non-null object
    Clicked on Ad               1000 non-null int64
    dtypes: float64(3), int64(3), object(4)
    memory usage: 78.2+ KB
```

Observation:
* We have both numerical and non-numerical columns to deal with

```python
#Describe numerical features
advert.describe()
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
      <th>Daily Time Spent on Site</th>
      <th>Age</th>
      <th>Area Income</th>
      <th>Daily Internet Usage</th>
      <th>Male</th>
      <th>Clicked on Ad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>65.000200</td>
      <td>36.009000</td>
      <td>55000.000080</td>
      <td>180.000100</td>
      <td>0.481000</td>
      <td>0.50000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15.853615</td>
      <td>8.785562</td>
      <td>13414.634022</td>
      <td>43.902339</td>
      <td>0.499889</td>
      <td>0.50025</td>
    </tr>
    <tr>
      <th>min</th>
      <td>32.600000</td>
      <td>19.000000</td>
      <td>13996.500000</td>
      <td>104.780000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>51.360000</td>
      <td>29.000000</td>
      <td>47031.802500</td>
      <td>138.830000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>68.215000</td>
      <td>35.000000</td>
      <td>57012.300000</td>
      <td>183.130000</td>
      <td>0.000000</td>
      <td>0.50000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>78.547500</td>
      <td>42.000000</td>
      <td>65470.635000</td>
      <td>218.792500</td>
      <td>1.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>91.430000</td>
      <td>61.000000</td>
      <td>79484.800000</td>
      <td>269.960000</td>
      <td>1.000000</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>

Observations:
* Males and females seem to be almost even
* Seems like we would have to include all of these features in our prediction

```python
#Describe non-numerical features
advert.describe(include=['O'])
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
      <th>Ad Topic Line</th>
      <th>City</th>
      <th>Country</th>
      <th>Timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>1000</td>
      <td>969</td>
      <td>237</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Multi-layered tangible portal</td>
      <td>Lisamouth</td>
      <td>Czech Republic</td>
      <td>2016-01-10 23:14:30</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

Observations:
* No feature seems to be worth adding in our prediction model. All will be dropped.

```python
#Lets drop non-numerical features
advert = advert.select_dtypes(exclude='object')
```

```python
#Check for null values in dataset
sns.heatmap(advert.isnull(),cmap='viridis',cbar=False,yticklabels=False)

    <matplotlib.axes._subplots.AxesSubplot at 0x16797fd3dd8>
```

![png](/projects/Analyzing_advertising_data/images/Analyzing_advertising_data_15_1.png)

```python
#Another way to check for null values in dataset
[col for col in advert.columns if advert[col].isnull().any()]

    []
```

Looks like there are no null values. Makes our life easier :)

### Exploratory data analysis

```python
advert['Age'].hist(bins=30)
plt.xlabel('Age')

    Text(0.5, 0, 'Age')
```

![png](/projects/Analyzing_advertising_data/images/Analyzing_advertising_data_19_1.png)

```python
sns.jointplot(x='Age',y='Area Income',data=advert,kind='kde',color='red')

    <seaborn.axisgrid.JointGrid at 0x16798305ac8>
```

![png](/projects/Analyzing_advertising_data/images/Analyzing_advertising_data_20_1.png)

```python
sns.jointplot('Age','Daily Internet Usage',advert,kind='hex',color='green')

    <seaborn.axisgrid.JointGrid at 0x16798422e80>
```

![png](/projects/Analyzing_advertising_data/images/Analyzing_advertising_data_21_1.png)

```python
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=advert)

    <seaborn.axisgrid.JointGrid at 0x1679a61ce80>
```

![png](/projects/Analyzing_advertising_data/images/Analyzing_advertising_data_22_1.png)

```python
sns.countplot(x='Male',hue='Clicked on Ad',data=advert)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    <matplotlib.legend.Legend at 0x1679a71f588>
```

![png](/projects/Analyzing_advertising_data/images/Analyzing_advertising_data_23_1.png)

`Male` feature is not the perfect predictor of our target.

```python
g = sns.FacetGrid(data=advert,row='Male',col='Clicked on Ad')
g.map(plt.hist,'Age')

    <seaborn.axisgrid.FacetGrid at 0x1679a78f748>
```

![png](/projects/Analyzing_advertising_data/images/Analyzing_advertising_data_25_1.png)

```python
g = sns.FacetGrid(data=advert,row='Male',col='Clicked on Ad')
g.map(plt.hist,'Daily Internet Usage')

    <seaborn.axisgrid.FacetGrid at 0x16798422ba8>
```

![png](/projects/Analyzing_advertising_data/images/Analyzing_advertising_data_26_1.png)

Seems like people with less internet usage have more click through rate.

### Define Pipeline

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
```

```python
X = advert.drop('Clicked on Ad',axis=1)
y = advert['Clicked on Ad']
```

```python
my_pipeline = make_pipeline(LogisticRegression(solver='liblinear'))
```

### Cross validation scores

```python
from sklearn.model_selection import cross_val_score
```

```python
cross_scores = cross_val_score(my_pipeline,X,y,cv=5,scoring='accuracy')
```

```python
cross_scores.mean()

    0.8959999999999999
```

We got a pretty good model, as our accuracy score is 89%. 

### Using XGBoost

```python
from xgboost import XGBClassifier,XGBRFClassifier
```

```python
from sklearn.model_selection import train_test_split
```

```python
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)
```

```python
model =XGBClassifier(n_estimators=1000,learning_rate=0.05,n_jobs=4)
```

```python
model.fit(X_train,y_train,early_stopping_rounds=5,eval_set=[(X_valid,y_valid)],verbose=False)

    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
           importance_type='gain', interaction_constraints='',
           learning_rate=0.05, max_delta_step=0, max_depth=6,
           min_child_weight=1, missing=nan, monotone_constraints='()',
           n_estimators=1000, n_jobs=4, num_parallel_tree=1,
           objective='binary:logistic', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
           validate_parameters=1, verbosity=None)
```

```python
predictions = model.predict(X_valid)
```

```python
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
```

```python
accuracy_score(y_valid,predictions)

    0.94
```

```python
print(classification_report(y_valid,predictions))

                  precision    recall  f1-score   support
    
               0       0.90      0.99      0.94       103
               1       0.99      0.89      0.93        97
    
       micro avg       0.94      0.94      0.94       200
       macro avg       0.95      0.94      0.94       200
    weighted avg       0.94      0.94      0.94       200
```

Great, using XGBoost classifier, we were able to achieve 94% accuracy. Let us now test using the `RandomForest` flavor of XGBoost.

```python
model = XGBRFClassifier (n_estimators=1000,learning_rate=0.05,n_jobs=4)
```

```python
model.fit(X_train,y_train,early_stopping_rounds=5,eval_set=[(X_valid,y_valid)],verbose=False)

    XGBRFClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bynode=0.8, colsample_bytree=1, gamma=0, gpu_id=-1,
            importance_type='gain', interaction_constraints='',
            learning_rate=0.05, max_delta_step=0, max_depth=6,
            min_child_weight=1, missing=nan, monotone_constraints='()',
            n_estimators=1000, n_jobs=4, num_parallel_tree=1000,
            objective='binary:logistic', random_state=0, reg_alpha=0,
            reg_lambda=1e-05, scale_pos_weight=1, subsample=0.8,
            tree_method='exact', validate_parameters=1, verbosity=None)
```

```python
predictions = model.predict(X_valid)
```

```python
accuracy_score(y_valid,predictions)

    0.96
```

```python
print(classification_report(y_valid,predictions))

                  precision    recall  f1-score   support
    
               0       0.93      1.00      0.96       103
               1       1.00      0.92      0.96        97
    
       micro avg       0.96      0.96      0.96       200
       macro avg       0.96      0.96      0.96       200
    weighted avg       0.96      0.96      0.96       200
```

Awesome, we were able to achieve even better results on this.
