# -*- coding: utf-8 -*-
import scrapy


class ExampleSpider(scrapy.Spider):
    name = 'example'
    allowed_domains = ['example.webscraping.com']
    start_urls = ['http://example.webscraping.com/']

    def parse(self, response):
        countries = response.xpath("//div[@id='results']/descendant::div")
        for country in countries:
            link = response.urljoin(country.xpath(".//a/@href").get())
            yield scrapy.Request(
                url=link,
                callback=self.parse_countries
            )
        next_page = response.urljoin(response.xpath("//div[@id='pagination']/a/@href").get())
        if next_page:
            yield scrapy.Request(
                url=next_page,
                callback=self.parse
            )

    def parse_countries(self, response):
        country_name = response.xpath("//tr[@id='places_country__row']/td[@class='w2p_fw']/text()").get()
        capital = response.xpath("//tr[@id='places_capital__row']/td[@class='w2p_fw']/text()").get()
        population = response.xpath("//tr[@id='places_population__row']/td[@class='w2p_fw']/text()").get()
        yield {
            'country_name': country_name,
            'country_capital': capital,
            'country_population': population
        }
