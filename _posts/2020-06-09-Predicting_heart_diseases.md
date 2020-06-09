---
title:  "Project: Predicting heart diseases"
date: 2020-06-09
categories: [Project]
tags: [machine learning,data science,python]
excerpt: "Predicting if a patient has a heart disease or not based on the given features. We will be using pipelines, cross_validation_scores and KNN model"
author_profile: true
mathjax: true
---

In this project, we are going to use KNN model to predict the heart disease. We are going to use this [dataset](https://www.kaggle.com/ronitf/heart-disease-uci) from Kaggle.

Highlights:
* Using StandardScaler to scale the features
* Defining Pipelines
* Using KNN model for predictions

```python
#Imports
import numpy as np
import pandas as pd
```

```python
#Load data
df = pd.read_csv('heart.csv')
```

```python
#Glimpse of data
df.head(3)
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
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
#Dropping the rows where target has null value
df.dropna(axis=0,subset=['target'],inplace=True)
```

```python
#Separate predictors and target
X = df.drop('target',axis=1)
y = df['target']
```

```python
#check for null values
[col for col in df.columns if df[col].isnull().any()]

    []
```

```python
#Imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
```

```python
#Define pipeline
my_pipline = Pipeline(steps=[
    ('scaler',StandardScaler()),
    ('model',KNeighborsClassifier())
])
```

```python
#Calculate cross validation scores
scores = cross_val_score(my_pipline,X,y,cv=5,scoring='accuracy')
```

```python
scores.mean()

    0.8150819672131148
```

We were able to achieve 81% accuracy with no parameter tuning. Let's try to tune the parameter `n_neighbors` and see what results can we achieve.

```python
mean_scores = {}
```

```python
for i in range(1,50):
    my_pipline = Pipeline(steps=[
        ('scaler',StandardScaler()),
        ('model',KNeighborsClassifier(n_neighbors=i))
    ])
    scores = cross_val_score(my_pipline,X,y,cv=5,scoring='accuracy')
    mean_scores[i] = scores.mean()
```

```python
# Finding the key with the best value
max(mean_scores,key=lambda x:mean_scores[x])

    28
```

```python
#Replugging that value into the model and re-calculating the cross-validation scores
my_pipline = Pipeline(steps=[
    ('scaler',StandardScaler()),
    ('model',KNeighborsClassifier(n_neighbors=28))
])
```

```python
scores = cross_val_score(my_pipline,X,y,cv=5,scoring='accuracy')
```

```python
scores.mean()

    0.8348633879781421
```

We were able to achieve the accuracy of 83.5% using parameter tuning.
