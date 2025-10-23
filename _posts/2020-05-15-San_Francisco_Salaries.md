---
title:  "Project: Analyzing San Francisco Salaries Data"
date: 2020-05-15
categories: [Project]
tags: [machine learning,data science,python]
excerpt: "Analyzing the SF employee's salaries data using Pandas, Numpy, Matplotlib, Seaborn and Sklearn"
author_profile: true
mathjax: true
---

In this project, we will analyze San Francisco employee's salaries data that is available on [Kaggle](https://www.kaggle.com/kaggle/sf-salaries). We will learn:
* Data exploration using Pandas
* Converting object columns to numeric
* Finding null values
* Imputing null values using both fillna method and SimpleImputer

```python
#Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
#Import data from the csv file
salaries = pd.read_csv('Salaries.csv')
```

```python
#Glimpse of data
salaries.head()
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
      <th>Id</th>
      <th>EmployeeName</th>
      <th>JobTitle</th>
      <th>BasePay</th>
      <th>OvertimePay</th>
      <th>OtherPay</th>
      <th>Benefits</th>
      <th>TotalPay</th>
      <th>TotalPayBenefits</th>
      <th>Year</th>
      <th>Notes</th>
      <th>Agency</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NATHANIEL FORD</td>
      <td>GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY</td>
      <td>167411</td>
      <td>0</td>
      <td>400184</td>
      <td>NaN</td>
      <td>567595.43</td>
      <td>567595.43</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GARY JIMENEZ</td>
      <td>CAPTAIN III (POLICE DEPARTMENT)</td>
      <td>155966</td>
      <td>245132</td>
      <td>137811</td>
      <td>NaN</td>
      <td>538909.28</td>
      <td>538909.28</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>ALBERT PARDINI</td>
      <td>CAPTAIN III (POLICE DEPARTMENT)</td>
      <td>212739</td>
      <td>106088</td>
      <td>16452.6</td>
      <td>NaN</td>
      <td>335279.91</td>
      <td>335279.91</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>CHRISTOPHER CHONG</td>
      <td>WIRE ROPE CABLE MAINTENANCE MECHANIC</td>
      <td>77916</td>
      <td>56120.7</td>
      <td>198307</td>
      <td>NaN</td>
      <td>332343.61</td>
      <td>332343.61</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>PATRICK GARDNER</td>
      <td>DEPUTY CHIEF OF DEPARTMENT,(FIRE DEPARTMENT)</td>
      <td>134402</td>
      <td>9737</td>
      <td>182235</td>
      <td>NaN</td>
      <td>326373.19</td>
      <td>326373.19</td>
      <td>2011</td>
      <td>NaN</td>
      <td>San Francisco</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

```python
#Data information
salaries.info()

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 148654 entries, 0 to 148653
    Data columns (total 13 columns):
    Id                  148654 non-null int64
    EmployeeName        148654 non-null object
    JobTitle            148654 non-null object
    BasePay             148049 non-null object
    OvertimePay         148654 non-null object
    OtherPay            148654 non-null object
    Benefits            112495 non-null object
    TotalPay            148654 non-null float64
    TotalPayBenefits    148654 non-null float64
    Year                148654 non-null int64
    Notes               0 non-null float64
    Agency              148654 non-null object
    Status              38119 non-null object
    dtypes: float64(3), int64(2), object(8)
    memory usage: 14.7+ MB
```

```python
#Before we start analyzing the data. Lets see if we have any null values anywhere. We might have to take care of them first.
plt.figure(figsize=(10,6))
sns.heatmap(salaries.isnull(),cmap='viridis',yticklabels=False,cbar=False)

    <matplotlib.axes._subplots.AxesSubplot at 0x242869b5b00>
```

![png](/projects/San_Francisco_Salaries/images/San_Francisco_Salaries_7_1.png)

```python
#Find the columns where null value exists and then count the null values
{col:salaries[col].isnull().sum() for col in salaries.columns if salaries[col].isnull().any()}

    {'BasePay': 605, 'Benefits': 36159, 'Notes': 148654, 'Status': 110535}
```

Looks like we have to drop the *Notes* and *Status* column as there are alot of null values.

```python
salaries.drop(['Notes','Status'],axis=1,inplace=True)
```

Let's now focus on the `BasePay` and `Benefits` column. We will convert both the columns to 'float64' type. However, both the columns contain 'Not Provided' value which we will convert to nan so that it will be treated with other nan values.

```python
salaries['BasePay'][salaries['BasePay']=='Not Provided'] = np.nan
```

```python
salaries['BasePay'] = pd.to_numeric(salaries['BasePay'])
```

```python
salaries['BasePay'].dtype

    dtype('float64')
```

Let's repeat the same for 'Benefits' column.

```python
salaries['Benefits'][salaries['Benefits']=='Not Provided'] = np.nan
```

```python
salaries['Benefits'] = pd.to_numeric(salaries['Benefits'])
```

```python
salaries['BasePay'].dtype

    dtype('float64')
```

Now that both the columns are numeric, we can take care of the null values.

```python
salaries['Benefits'].fillna(value=0,inplace=True)
```

We will fill the null values of BasePay column with the mean of that column. However, i want to show you another method of taking care of null values i.e. `SimpleImputer`

```python
from sklearn.impute import SimpleImputer
```

```python
si = SimpleImputer(strategy='mean')
```

```python
salaries['BasePay'] = pd.DataFrame(si.fit_transform(np.array(salaries['BasePay']).reshape(-1,1)))
```

```python
#Lets check if we have any null values in BasePay column now
salaries['BasePay'].isnull().any()

    False
```

Now that we have taken care of the null values, lets explore the data.

```python
#Avg BasePay
salaries['BasePay'].mean()

    66325.44884050524
```

```python
#Job title of DAVID FRANKLIN
salaries[salaries['EmployeeName'] == 'DAVID FRANKLIN']['JobTitle']

    14    BATTALION CHIEF, (FIRE DEPARTMENT)
    Name: JobTitle, dtype: object
```

```python
#How much does DAVID FRANKLIN make
salaries[salaries['EmployeeName'] == 'DAVID FRANKLIN']['TotalPayBenefits']

    14    286347.05
    Name: TotalPayBenefits, dtype: float64
```

```python
#Name of highest paid person
salaries[salaries['TotalPayBenefits']==salaries['TotalPayBenefits'].max()]['EmployeeName']

    0    NATHANIEL FORD
    Name: EmployeeName, dtype: object
```

```python
#Name and pay of lowest paid person
salaries[salaries['TotalPayBenefits']==salaries['TotalPayBenefits'].min()][['EmployeeName','TotalPayBenefits']]
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
      <th>EmployeeName</th>
      <th>TotalPayBenefits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>148653</th>
      <td>Joe Lopez</td>
      <td>-618.13</td>
    </tr>
  </tbody>
</table>
</div>

```python
#Avg BasePay of all employees per year from 2011-2014
salaries.groupby(by='Year').mean()['BasePay']

    Year
    2011    63595.956517
    2012    65436.406857
    2013    69576.866579
    2014    66564.396851
    Name: BasePay, dtype: float64

```

```python
#Number of Unique job titles
salaries['JobTitle'].nunique()

    2159
```

```python
#Top 5 most common jobs
salaries['JobTitle'].value_counts().head()

    Transit Operator                7036
    Special Nurse                   4389
    Registered Nurse                3736
    Public Svc Aide-Public Works    2518
    Police Officer 3                2421
    Name: JobTitle, dtype: int64
```

```python
#Number of Job titles with only one occurence in 2013
sum(salaries[salaries['Year']==2013]['JobTitle'].value_counts()==1)

    202
```

```python
#To view those 202 Job titles
salaries.groupby(by=['Year','JobTitle']).count().loc[2013,'Id'][salaries.groupby(by=['Year','JobTitle']).count().loc[2013,'Id']==1]

    JobTitle
    Acupuncturist                     1
    Adm, SFGH Medical Center          1
    Administrative Analyst I          1
    Administrative Analyst II         1
    Administrator, DPH                1
    Airport Communications Officer    1
    Airport Mechanical Maint Sprv     1
    Animal Care Asst Supv             1
    Animal Care Supervisor            1
    Animal Control Supervisor         1
    Arborist Technician Supv II       1
    Area Sprv Parks, Squares & Fac    1
    Asphalt Plant Supervisor 1        1
    Assessor                          1
    Assistant Director, Probate       1
    Assistant Industrial Hygienist    1
    Assistant Inspector               1
    Assistant Inspector 2             1
    Assistant Law Librarian           1
    Assistant Power House Operator    1
    Assistant Sheriff                 1
    Assoc Musm Cnsrvt, AAM            1
    Asst Chf, Bur Clm Invest&Admin    1
    Asst Dir of Clinical Svcs 1       1
    Asst Dir, Log Cabin Rnch          1
    Asst Director, Juvenile Hall      1
    Asst Superintendent Rec           1
    Attorney, Tax Collector           1
    Auto Body & Fender Wrk Sprv 1     1
    Baker                             1
                                     ..
    Special Assistant 21              1
    Specialist in Aging 2             1
    Sprv Adult Prob Ofc (SFERS)       1
    Sr Employee Asst Counselor        1
    Sr General Utility Mechanic       1
    Sr Light Rail Veh Equip Eng       1
    Sr Medical Transcriber Typist     1
    Sr Opers Mgr                      1
    Statistician                      1
    Sup Ct Admin Secretary            1
    Sup Welfare Fraud Investigator    1
    Supervising Parts Storekeeper     1
    Supply Room Attendant             1
    Telecommunications Tech Supv      1
    Track Maint Supt, Muni Railway    1
    Traf Signal Electrician Sup II    1
    Traffic Sign Manager              1
    Traffic Signal Operator           1
    Training Coordinator              1
    Training Technician               1
    Transit Paint Shop Sprv1          1
    Treasurer                         1
    Trnst Power Line Wrk Sprv 2       1
    Undersheriff                      1
    Vet Laboratory Technologist       1
    Victim & Witness Technician       1
    Water Meter Shop Supervisor 1     1
    Wharfinger 1                      1
    Window Cleaner Supervisor         1
    Wire Rope Cable Maint Sprv        1
    Name: Id, Length: 202, dtype: int64
```

```python
#How many people have the word Chief in their job title
len(salaries[salaries['JobTitle'].apply(lambda x:'chief' in x.lower())])

    627
```

```python
#Is there a relation between length of Jobtitle and Salary
salaries['title_length'] = salaries['JobTitle'].apply(len)
```

```python
salaries[['title_length','TotalPayBenefits']].corr()
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
      <th>title_length</th>
      <th>TotalPayBenefits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>title_length</th>
      <td>1.000000</td>
      <td>-0.036878</td>
    </tr>
    <tr>
      <th>TotalPayBenefits</th>
      <td>-0.036878</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

Seems there is no correlation between the two variables.
