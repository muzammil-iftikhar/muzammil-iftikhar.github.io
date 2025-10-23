---
title:  "Project: Analyzing Stocks"
date: 2020-05-16
categories: [Project]
tags: [data science,python,finance,stocks]
excerpt: "Analyzing the stocks of four different companies from year 2006 to 2015"
author_profile: true
mathjax: true
---

In this project, we will focus on exploratory data analysis of stock prices for following companies from 2006 to 2015.
* Amazon
* Microsoft
* Bank of America
* Citibank

```python
#Imports
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
```

```python
#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
#Start and end period
start = dt.datetime(2006,1,1)
end = dt.datetime(2016,1,1)
```

```python
#Tickers of the companies
tickers = ['AMZN','MSFT','BAC','C']
```

```python
#Fetch data from yahoo stocks using DataReader
amazon = web.DataReader('AMZN',data_source='yahoo',start=start,end=end)
microsoft = web.DataReader('MSFT',data_source='yahoo',start=start,end=end)
bank_of_america = web.DataReader('BAC',data_source='yahoo',start=start,end=end)
citigroup = web.DataReader('C',data_source='yahoo',start=start,end=end)
```

```python
#Concating all the data frames of different companies into one large data frame. 
stock_data = pd.concat([amazon,microsoft,bank_of_america,citigroup],axis=1,keys=tickers)
```

```python
#Adding column names
stock_data.columns.names = ['Tickers','Stock Info']
```

```python
#Glimpse of data
stock_data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table class="dataframe">
  <thead>
    <tr>
      <th>Tickers</th>
      <th colspan="6" halign="left">AMZN</th>
      <th colspan="6" halign="left">MSFT</th>
      <th colspan="6" halign="left">BAC</th>
      <th colspan="6" halign="left">C</th>
    </tr>
    <tr>
      <th>Stock Info</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2006-01-03</th>
      <td>47.849998</td>
      <td>46.250000</td>
      <td>47.470001</td>
      <td>47.580002</td>
      <td>7582200</td>
      <td>47.580002</td>
      <td>27.000000</td>
      <td>26.10</td>
      <td>26.250000</td>
      <td>26.840000</td>
      <td>79973000.0</td>
      <td>19.602528</td>
      <td>47.180000</td>
      <td>46.150002</td>
      <td>46.919998</td>
      <td>47.080002</td>
      <td>16296700.0</td>
      <td>35.298687</td>
      <td>493.799988</td>
      <td>481.100006</td>
      <td>490.000000</td>
      <td>492.899994</td>
      <td>1537600.0</td>
      <td>440.882477</td>
    </tr>
    <tr>
      <th>2006-01-04</th>
      <td>47.730000</td>
      <td>46.689999</td>
      <td>47.490002</td>
      <td>47.250000</td>
      <td>7440900</td>
      <td>47.250000</td>
      <td>27.080000</td>
      <td>26.77</td>
      <td>26.770000</td>
      <td>26.969999</td>
      <td>57975600.0</td>
      <td>19.697485</td>
      <td>47.240002</td>
      <td>46.450001</td>
      <td>47.000000</td>
      <td>46.580002</td>
      <td>17757900.0</td>
      <td>34.923801</td>
      <td>491.000000</td>
      <td>483.500000</td>
      <td>488.600006</td>
      <td>483.799988</td>
      <td>1870900.0</td>
      <td>432.742950</td>
    </tr>
    <tr>
      <th>2006-01-05</th>
      <td>48.200001</td>
      <td>47.110001</td>
      <td>47.160000</td>
      <td>47.650002</td>
      <td>5417200</td>
      <td>47.650002</td>
      <td>27.129999</td>
      <td>26.91</td>
      <td>26.959999</td>
      <td>26.990000</td>
      <td>48245500.0</td>
      <td>19.712091</td>
      <td>46.830002</td>
      <td>46.320000</td>
      <td>46.580002</td>
      <td>46.639999</td>
      <td>14970700.0</td>
      <td>34.968796</td>
      <td>487.799988</td>
      <td>484.000000</td>
      <td>484.399994</td>
      <td>486.200012</td>
      <td>1143100.0</td>
      <td>434.889679</td>
    </tr>
    <tr>
      <th>2006-01-06</th>
      <td>48.580002</td>
      <td>47.320000</td>
      <td>47.970001</td>
      <td>47.869999</td>
      <td>6152900</td>
      <td>47.869999</td>
      <td>27.000000</td>
      <td>26.49</td>
      <td>26.889999</td>
      <td>26.910000</td>
      <td>100963000.0</td>
      <td>19.653666</td>
      <td>46.910000</td>
      <td>46.349998</td>
      <td>46.799999</td>
      <td>46.570000</td>
      <td>12599800.0</td>
      <td>34.916302</td>
      <td>489.000000</td>
      <td>482.000000</td>
      <td>488.799988</td>
      <td>486.200012</td>
      <td>1370200.0</td>
      <td>434.889679</td>
    </tr>
    <tr>
      <th>2006-01-09</th>
      <td>47.099998</td>
      <td>46.400002</td>
      <td>46.549999</td>
      <td>47.080002</td>
      <td>8943100</td>
      <td>47.080002</td>
      <td>27.070000</td>
      <td>26.76</td>
      <td>26.930000</td>
      <td>26.860001</td>
      <td>55625000.0</td>
      <td>19.617136</td>
      <td>46.970001</td>
      <td>46.360001</td>
      <td>46.720001</td>
      <td>46.599998</td>
      <td>15619400.0</td>
      <td>34.938789</td>
      <td>487.399994</td>
      <td>483.000000</td>
      <td>486.000000</td>
      <td>483.899994</td>
      <td>1680700.0</td>
      <td>432.832489</td>
    </tr>
  </tbody>
</table>
</div>

```python
#Max close price for each company's stock
stock_data.xs(key='Close',level='Stock Info',axis=1).max()

    Tickers
    AMZN    693.969971
    MSFT     56.549999
    BAC      54.900002
    C       564.099976
    dtype: float64
```

```python
#Return of each company
returns = pd.DataFrame()
for ticker in tickers:
    returns[ticker] = stock_data[ticker]['Close'].pct_change()
```

```python
returns.head()
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
      <th>AMZN</th>
      <th>MSFT</th>
      <th>BAC</th>
      <th>C</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2006-01-03</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2006-01-04</th>
      <td>-0.006936</td>
      <td>0.004843</td>
      <td>-0.010620</td>
      <td>-0.018462</td>
    </tr>
    <tr>
      <th>2006-01-05</th>
      <td>0.008466</td>
      <td>0.000742</td>
      <td>0.001288</td>
      <td>0.004961</td>
    </tr>
    <tr>
      <th>2006-01-06</th>
      <td>0.004617</td>
      <td>-0.002964</td>
      <td>-0.001501</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2006-01-09</th>
      <td>-0.016503</td>
      <td>-0.001858</td>
      <td>0.000644</td>
      <td>-0.004731</td>
    </tr>
  </tbody>
</table>
</div>

```python
sns.pairplot(returns)

    <seaborn.axisgrid.PairGrid at 0x1877dbb83c8>
```

![png](/projects/Finance_project/images/Finance_project_14_1.png)

```python
#Worst return date for each company
returns.idxmin()

    AMZN   2006-07-26
    MSFT   2009-01-22
    BAC    2009-01-20
    C      2009-02-27
    dtype: datetime64[ns]
```

```python
#Best return date for each company
returns.idxmax()

    AMZN   2007-04-25
    MSFT   2008-10-13
    BAC    2009-04-09
    C      2008-11-24
    dtype: datetime64[ns]
```

```python
#Standard deviation of returns
returns.std()

    AMZN    0.026638
    MSFT    0.017764
    BAC     0.036647
    C       0.038672
    dtype: float64
```

```python
#Standard deviation of returns for year 2015
returns.loc['2015-01-01':'2015-12-31'].std()

    AMZN    0.021147
    MSFT    0.017801
    BAC     0.016163
    C       0.015289
    dtype: float64
```

```python
#Lets create distribution plot of the stock returns of Amazon for year 2015
sns.distplot(returns.loc['2015-01-01':'2015-12-31']['AMZN'],bins=100)

    <matplotlib.axes._subplots.AxesSubplot at 0x1877f3e4588>
```

![png](/projects/Finance_project/images/Finance_project_19_1.png)

```python
#Lets create distribution plot of the stock returns of Citigroup for year 2008
sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C'],bins=100)

    <matplotlib.axes._subplots.AxesSubplot at 0x187009ce518>
```

![png](/projects/Finance_project/images/Finance_project_20_1.png)

```python
#Close price of each company throughout the entire time period of 2006-2015
stock_data.xs(key='Close',axis=1,level='Stock Info').plot(figsize=(10,5))
plt.legend()

    <matplotlib.legend.Legend at 0x1877fdf1278>
```

![png](/projects/Finance_project/images/Finance_project_21_1.png)

```python
#30 day moving average of Amazon for year 2015
stock_data.loc['2015-1-1':'2015-12-31','AMZN']['Close'].rolling(30).mean().plot(figsize=(10,5),label='MA30')
stock_data.loc['2015-1-1':'2015-12-31','AMZN']['Close'].plot(label='Close')
plt.tight_layout()
plt.legend()

    <matplotlib.legend.Legend at 0x1870483e048>
```

![png](/projects/Finance_project/images/Finance_project_22_1.png)

```python
#30 day moving average of Citigroup for year 2008
stock_data.loc['2008-1-1':'2008-12-31','C']['Close'].rolling(30).mean().plot(figsize=(10,5),label='MA30')
stock_data.loc['2008-1-1':'2008-12-31','C']['Close'].plot(label='Close')
plt.tight_layout()
plt.legend()

    <matplotlib.legend.Legend at 0x18704839898>
```

![png](/projects/Finance_project/images/Finance_project_23_1.png)

```python
#Heatmap for the relation between close prices of different stocks
sns.heatmap(stock_data.xs(key='Close',axis=1,level='Stock Info').corr(),cmap='magma_r',annot=True)

    <matplotlib.axes._subplots.AxesSubplot at 0x18704ae9a58>
```

![png](/projects/Finance_project/images/Finance_project_24_1.png)
