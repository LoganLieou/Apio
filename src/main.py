import scrapy

class MedSpider(scrapy.Spider):
   name = 'medspider'
   start_urls = ["https://pubmed.ncbi.nlm.nih.gov/?term=deep%20learning&page=" + str(x) for x in range(1, 30374)]

   def parse(self, response):
      for item in response.css('div.docsum-content'):
        yield {
            'title': item.css('a.docsum-title::text').getall(),
            'content': item.css('div.full-view-snippet::text').getall()
         }
