---
title:  "Webscraping sites with infinite scroll"
date: 2020-06-29
categories: [Tutorial]
tags: [machine learning, data science, python, webscraping]
excerpt: "Learn how to scrape websites that have infinite scrolling"
author_profile: true
mathjax: true
---

In this tutorial, we are going to scrape a demo site that uses scrolling to fetch new data. We will be using [this site](http://quotes.toscrape.com/) for the purpose. We will be scraping the quotes and author names from this site.

We won't be scraping this site using traditional method, like sending the request to the site and extracting the *quote* and *author* tag from the response. Instead, we will be using the API to get what we want. Open the developer tools in chrome *ctrl+shift+i*. Select the network tab and select *XHR*. Now, once you scroll the site to the bottom, it will send the GET request to the api to fetch data. We will use this URL to fetch our results.

```python
#Start a new scrapy project
scrapy startproject toscrape
```

```python
#generate a new spider inside this  project
cd toscrape
scrapy genspider quote quotes.toscrape.com
```

```python
#api url, we will be sending the request to
start_urls = ['http://quotes.toscrape.com/api/quotes?page=1']
```

Once, a request is sent on the above URL by scrapy. Response will be in JSON. Hence, we need to convert that to python dictionary before we can use that.

```python
import json
```

```python
#convert JSON to python dictionary
result = json.loads(response.text)
```

```python
#get quotes key from result
quotes = result['quotes']
```

```python
#to actually get the quotes and author name
for quote in quotes:
    yield{
        'quote': quote['text'],
        'author': quote['author']['name']
    }
```

```python
#Now we will handle pagination
i = 2
next_page = result['has_next']
if next_page:
    next_link = f'http://quotes.toscrape.com/api/quotes?page={self.i}'
    self.i = self.i + 1
    yield scrapy.Request(
        url=next_link,
        callback=self.parse
    )
```

```python
#To get the data in the form of csv
scrapy crawl quote -o dataset.csv
```

You can get the complete code [here](/projects/Scraping_infinite_scroll/quote.py).
