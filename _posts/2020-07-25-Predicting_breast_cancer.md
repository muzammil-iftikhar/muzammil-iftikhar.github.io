---
title:  "Project: Predicting breast cancer"
date: 2020-07-25
categories: [Project]
tags: [machine learning, data science, python]
excerpt: "Predicting the breast tumor as either malignant or benign based on given features. We will be using SVM model and GridSearchCV"
author_profile: true
mathjax: true
---
In this project, we are going to use the built-in breast cancer dataset to predict whether the tumor is *malignant* or *benign*.

Highlights:
* Using Support vector machine model for predictions
* Using GridSearchCV to tune hyper-parameters

```python
# Imports
import pandas as pd
from sklearn.datasets import load_breast_cancer
```

```python
cancer = load_breast_cancer()
```

```python
# Creating data frame
cancer_features = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
```

```python
# Glimpse of data
cancer_features.head(2)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.8</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.6</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.9</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.8</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 30 columns</p>
</div>

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
```

```python
# Defining features X and target y
X = cancer_features
y = cancer['target']
# Creating train/test split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=1)
```

First we are going to train / validate the model without hyperparameter tuning. This way we will be better able to understand the importance of GridSearchCV specially with Support Vector Machine model.

```python
# Instantiate model
model = SVC()
```

```python
model.fit(X_train, y_train)

    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)
```


```python
predictions = model.predict(X_valid)
```

```python
from sklearn.metrics import classification_report
```

```python
print(classification_report(y_valid, predictions))

                  precision    recall  f1 - score   support

               0       0.00      0.00      0.00        42
               1       0.63      1.00      0.77        72
    
       micro avg       0.63      0.63      0.63       114
       macro avg       0.32      0.50      0.39       114
    weighted avg       0.40      0.63      0.49       114
```

You can see how the SVM model predicted everything in the benign class. Let us now use GridSearchCV to tune hyperparameters and see the difference in our accuracy scores.

```python
from sklearn.model_selection import GridSearchCV
```

```python
# Create parameters dictionary with different values to train the model on
param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
```

```python
# Instantiate the grid
grid = GridSearchCV(SVC(),param_grid,verbose=3)
```

```python
# Fit the grid
grid.fit(X_train,y_train)

    GridSearchCV(cv='warn', error_score='raise-deprecating',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=3)
```

```python
# Check the best values of parameters
grid.best_params_

    {'C': 10, 'gamma': 0.0001}
```

```python
# Generate predictions using those tuned hyperparameteric values
grid_predictions = grid.predict(X_valid)
```

```python
print(classification_report(y_valid,grid_predictions))

                  precision    recall  f1-score   support
    
               0       0.97      0.90      0.94        42
               1       0.95      0.99      0.97        72
    
       micro avg       0.96      0.96      0.96       114
       macro avg       0.96      0.95      0.95       114
    weighted avg       0.96      0.96      0.96       114
```

You can clearly see the difference, now our SVM model performed so well with an accuracy of around 96%.
