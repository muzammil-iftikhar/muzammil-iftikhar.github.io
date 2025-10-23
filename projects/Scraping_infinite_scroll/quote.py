# -*- coding: utf-8 -*-
import scrapy
import json


class QuoteSpider(scrapy.Spider):
    name = 'quote'
    allowed_domains = ['quotes.toscrape.com']
    start_urls = ['http://quotes.toscrape.com/api/quotes?page=1']
    i = 2

    def parse(self, response):
        result = json.loads(response.text)
        quotes = result['quotes']
        for quote in quotes:
            yield{
                'quote': quote['text'],
                'author': quote['author']['name']
            }
        next_page = result['has_next']
        if next_page:
            next_link = f'http://quotes.toscrape.com/api/quotes?page={self.i}'
            self.i = self.i + 1
            yield scrapy.Request(
                url=next_link,
                callback=self.parse
            )
