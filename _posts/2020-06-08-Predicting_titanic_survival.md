---
title:  "Project: Predicting Titanic survival"
date: 2020-06-08
categories: [Project]
tags: [machine learning,data science,python]
excerpt: "Predicting who were more likely to survive on Titanic based on the given features. We will be using pipelines, cross_validation_scores, LogisticRegression and XGBoost"
author_profile: true
mathjax: true
---

In this project, we will be working with the very famous [Titanic](https://www.kaggle.com/c/titanic) dataset from Kaggle.

Highlights:
* Exploratory data analysis
* SimpleImputer to impute null values
* OneHotEncoder to encode categorical variables
* Defining pipelines
* Cross validation scores
* LogisticRegression model
* RandomForestClassifier model
* XGBClassifier model
* XGBRFClassifier model

```python
#Imports
import numpy as np
import pandas as pd
```

```python
#Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
%matplotlib inline
sns.set()
```

```python
#Data loading
titanic = pd.read_csv('train.csv')
```

```python
#Glimpse of data
titanic.head(2)
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>

```python
#drop the rows where target has missing values
titanic.dropna(axis=0,subset=['Survived'],inplace=True)
```

```python
#Data information
titanic.info()

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 90.5+ KB
```

We have both numerical and non-numerical data to deal with. We would have to convert the categorical features to numerical ones before we could fit our model.

```python
#Describe numerical features
titanic.describe()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>

Observations:
* PassengerId holds not significant meaning and will be dropped.
* Pclass is an ordinal variable with most of the people in 3rd class
* Age seems to have null values, we will have to fill them in with some value
* SibSp is also an ordinal variable with most of the people travelling without siblings or spouses
* Parch is also an ordinal variable with most of the people travelling without their parents or children
* Max fare seems an outlier

```python
#Describe categorical features
titanic.describe(include=['O'])
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
      <th>Name</th>
      <th>Sex</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>891</td>
      <td>2</td>
      <td>681</td>
      <td>147</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Thomas, Master. Assad Alexander</td>
      <td>male</td>
      <td>347082</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>577</td>
      <td>7</td>
      <td>4</td>
      <td>644</td>
    </tr>
  </tbody>
</table>
</div>

Observations:
* We can assume that name holds no significance in the survival of a passenger, hence we will drop this column
* Sex will be included in our predictions
* Looks like some people were sharing tickets. We will drop this column as well as it doesn't show any significance
* Cabin seems to have alot of null values. We will drop it as well
* Embarked will be used in predictions

### Exploratory data analysis

```python
#Lets check for null values
plt.figure(figsize=(10,5))
sns.heatmap(titanic.isnull(),cbar=False,yticklabels=False,cmap='viridis')

    <matplotlib.axes._subplots.AxesSubplot at 0x28dd1756940>
```

![png](/projects/Predicting_titanic_survival/images/Predicting_titanic_survival_15_1.png)

```python
[col for col in titanic.columns if titanic[col].isnull().any()]

    ['Age', 'Cabin', 'Embarked']
```

```python
#We will drop cabin column as it has alot of null values. Dropping all unnecessary columns
titanic.drop(['Cabin','PassengerId','Name','Ticket'],axis=1,inplace=True)
```

We will impute the `Age` and `Embarked` columns later.

```python
sns.countplot(x=titanic['Survived'],data=titanic,hue='Sex')

    <matplotlib.axes._subplots.AxesSubplot at 0x28dd1ab0978>
```

![png](/projects/Predicting_titanic_survival/images/Predicting_titanic_survival_19_1.png)

Looks like there were more female survivors.

```python
sns.countplot(x=titanic['Survived'],data=titanic,hue='Embarked')

    <matplotlib.axes._subplots.AxesSubplot at 0x28dd1b03278>
```

![png](/projects/Predicting_titanic_survival/images/Predicting_titanic_survival_21_1.png)


```python
sns.countplot(x=titanic['Survived'],data=titanic,hue='Pclass')

    <matplotlib.axes._subplots.AxesSubplot at 0x28dd1b69470>
```

![png](/projects/Predicting_titanic_survival/images/Predicting_titanic_survival_22_1.png)

Looks like, people from 3rd class died more. Also, survival rate of 1st class is, somewhat, more than the other classes.

```python
sns.distplot(titanic['Age'].dropna(),kde=False,color='darkred',bins=30)

    <matplotlib.axes._subplots.AxesSubplot at 0x28dd1bd1c18>
```

![png](/projects/Predicting_titanic_survival/images/Predicting_titanic_survival_24_1.png)

```python
g = sns.FacetGrid(titanic,col='Survived',row='Sex')
g.map(plt.hist,'Age')

    <seaborn.axisgrid.FacetGrid at 0x28dd1add160>
```

![png](/projects/Predicting_titanic_survival/images/Predicting_titanic_survival_25_1.png)

More infants survived. Also, 80 years old guy survived.

```python
g = sns.FacetGrid(titanic,col='Survived',row='Pclass')
g.map(sns.countplot,'Embarked')

    <seaborn.axisgrid.FacetGrid at 0x28dd1c6df60>
```

![png](/projects/Predicting_titanic_survival/images/Predicting_titanic_survival_27_2.png)

```python
titanic['Fare'].hist(bins=30,color='green')
    <matplotlib.axes._subplots.AxesSubplot at 0x28dd2155c50>
```

![png](/projects/Predicting_titanic_survival/images/Predicting_titanic_survival_28_1.png)

### Pipelines

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
```

```python
#Separate target from predictors
X = titanic.drop('Survived',axis=1)
y = titanic['Survived']
```

```python
#Lets separate the numerical and categorical columns
num_cols = [col for col in X.columns if titanic[col].dtype in ['int64','float64']]
cat_cols = [col for col in X.columns if titanic[col].dtype == 'object']
```

```python
#Numerical transformer
num_transformer = SimpleImputer(strategy='mean')
```

```python
#Categorical transformer
cat_transformer = Pipeline(steps=[
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore',sparse=False))
])
```

```python
preprocessor = ColumnTransformer(transformers=[
    ('num',num_transformer,num_cols),
    ('cat',cat_transformer,cat_cols)
])
```

### Get Model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
```

```python
model1 = LogisticRegression(solver='liblinear',random_state=1)
model2 = RandomForestClassifier(n_estimators=250,random_state=1)
```

```python
#Model1 pipeline
final_pipeline_1 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model1)
])
```

```python
#Model2 pipeline
final_pipeline_2 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model2)
])
```

### Cross-Validation scores

```python
from sklearn.model_selection import cross_val_score
```

```python
scores_1 = cross_val_score(final_pipeline_1,X,y,cv=5,scoring='accuracy')
```

```python
scores_1.mean()

    0.7912852282814269
```

```python
scores_2 = cross_val_score(final_pipeline_2,X,y,cv=5,scoring='accuracy')
```

```python
scores_2.mean()

    0.8115290623015141
```

We have achieved an accuracy of around 81% using `RandomForest` model.

### XGBoost

```python
from xgboost import XGBClassifier,XGBRFClassifier
from sklearn.model_selection import train_test_split
```

```python
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)
```

```python
model_XGBClassifier = XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=4)
```

```python
model_XGBRClassifier = XGBRFClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=4)
```

```python
#Pipeline #1
pipeline_XGB = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model_XGBClassifier)
])
```

```python
#Pipeline #2
pipeline_XGBRF = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model_XGBRClassifier)
])
```

```python
params = {'model__early_stopping_rounds':5,
          'model__verbose':False,
          'model__eval_set':[(X_valid,y_valid)]}
```

```python
scores_XGB = cross_val_score(pipeline_XGB,X_train,y_train,cv=5,scoring='accuracy')
```

```python
scores_XGB.mean()

    0.8020450232017508
```

```python
scores_XGBRF = cross_val_score(pipeline_XGBRF,X_train,y_train,cv=5,scoring='accuracy')
```

```python
scores_XGBRF.mean()

    0.8356706923083909
```

Using the `RandomForest` flavor of XGBoost, we were able to achieve the accuracy of around 83.5%.
