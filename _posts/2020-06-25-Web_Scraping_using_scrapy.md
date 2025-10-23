---
title:  "Webscraping using scrapy"
date: 2020-06-25
categories: [Tutorial]
tags: [machine learning, data science, python, webscraping]
excerpt: "Learn and practice webscraping using scrapy"
author_profile: true
mathjax: true
---

In both Data science and Machine learning, a user would need a large amount of data to analyze, to train/test the model. And usually you won't be presented with the prepared dataset in a plate like you get from Kaggle. So, webscraping might come in handy. It's a knowledge that as a data scientist you should be aware of.  
Today, we are going to scrape a website using *scrapy*. We will be scraping [this](http://example.webscraping.com/) website.

Highlights:
* We will be using the basic template of scrapy to crawl the website
* We will be learning how to follow links
* We will be learning how to paginate while scraping
* We will be using Xpath instead of CSS selectors to scrape the elements

First things first, as a basic ethic and moral value, **ALWAYS** respect the rules of robots.txt while scraping.

Let's start, we will first send the initial request to the website and then we will visit each and every country url and scrape the following from there:
* Country name
* Capital
* Population

Based on the concepts, you can scrape anything you want. Then, we will move on to the next page and repeat those steps until we hit the last page.

```python
#Initialize the project
scrapy startproject exampleweb
```

```python
#Initialize the spider
scrapy genspider example example.webscraping.com
```

```python
#Xpath we will use to scrape the countries div
countries = response.xpath("//div[@id='results']/descendant::div")
```

```python
#We will then get the url of each country from <a> tag
for country in countries:
    link = response.urljoin(country.xpath(".//a/@href")).get()
```

```python
#We will then send the request to the scraped URLs
yield scrapy.Request(
    url=link,
    callback=self.parse_countries)
```

```python
#We will create a new class self.parse_countries to capture the response and get what we want from it
def parse_countries(self, response):
country_name = response.xpath("//tr[@id='places_country__row']/td[@class='w2p_fw']/text()".get())
capital = response.xpath("//tr[@id='places_capital__row']/td[@class='w2p_fw']/text()").get()
population = response.xpath("//tr[@id='places_population__row']/td[@class='w2p_fw']/text()").get()
yield {
    'country_name': country_name,
    'country_capital': capital,
    'country_population': population
}
```

```python
#Then we will handle the pagination and keep on collecting our desired data until we hit the last page
next_page = response.urljoin(response.xpath("//div[@id='pagination']/a/@href").get())
if next_page:
    yield scrapy.Request(
        url=next_page,
        callback=self.parse
    )
```

```python
#In the end, we will run the following command to get all the data in a file 'dataset.json'
scrapy crawl example -o dataset.json
#You can also save the file as csv
scrapy crawl example -o dataset.csv
```

The complete code of above can be found here [example.py](/projects/Webscraping_example/example.py)

Now that we have our data, we will use *pd.read_csv* or *pd.read_json* to import our data as dataframe.

```python
import pandas as pd
```

```python
pd.read_json('dataset.json').head(3)
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
      <th>country_capital</th>
      <th>country_name</th>
      <th>country_population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>St. John's</td>
      <td>Antigua and Barbuda</td>
      <td>86,754</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td>Antarctica</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Valley</td>
      <td>Anguilla</td>
      <td>13,254</td>
    </tr>
  </tbody>
</table>
</div>
