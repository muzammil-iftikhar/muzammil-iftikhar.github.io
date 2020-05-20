---
title:  "Effect of an outlier on Regression Line"
date: 2020-05-20
categories: [Reading]
tags: [machine learning, data science, python, statistics, regression, outlier]
excerpt: "Visualize how an outlier effects the regression line"
author_profile: true
mathjax: true
---

Just a small article to show the impact of an outlier in the direction of your dependent variable.

```python
# Necessary imports
import pandas as pd
import seaborn as sns
```

```python
# Lets define a small random dataset to prove our point
df = pd.DataFrame({'x': [1, 4, 5, 8, 10, 13], 'y': [3, 5, 8, 10, 15, 20]})
```

```python
df
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
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>


```python
# Lets plot the regression line b/w x and y where x is your independent variable and y is the dependent variable
sns.lmplot('x', 'y', df)
  <seaborn.axisgrid.FacetGrid at 0x1dd68a89ba8 >
```

![png](/projects/Effect_of_outlier_on_regression_line/images/Effect_of_outlier_on_regression_line_5_1.png)

```python
# Lets check the correlation b/w x and y
df.corr()
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
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>x</th>
      <td>1.000000</td>
      <td>0.981795</td>
    </tr>
    <tr>
      <th>y</th>
      <td>0.981795</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

We can see that the correlation is so strong between x and y. Let us now place an outlier in the direction of the dependent variable and see the effect of it on the correlation value

```python
# Lets place an outlier in the direction of x-axis
df = pd.DataFrame({'x': [1, 4, 5, 8, 10, 13, 100],
                   'y': [3, 5, 8, 10, 15, 20, 5]})
```

```python
df
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
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13</td>
      <td>20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>

```python
sns.lmplot('x', 'y', df)
  <seaborn.axisgrid.FacetGrid at 0x1dd68ad92b0 >
```

![png](/projects/Effect_of_outlier_on_regression_line/images/Effect_of_outlier_on_regression_line_10_1.png)


```python
df.corr()
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
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>x</th>
      <td>1.000000</td>
      <td>-0.211966</td>
    </tr>
    <tr>
      <th>y</th>
      <td>-0.211966</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

The relation has gone from a very strong positive relation to a very weak negative relation. Hence, it is always a good idea to investigate those outliers in case of small datasets, they may point to a potential opportunity or in worst case, just drop them altogether as they can adversely affect the performance of your regression model.
