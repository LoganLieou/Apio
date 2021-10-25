import scrapy

class TitleSpider(scrapy.Spider):
   name = 'medspider1'
   start_urls = ["https://pubmed.ncbi.nlm.nih.gov/?term=deep%20learning&page=" + str(x) for x in range(1, 303)]

   def parse(self, response):
      for item in response.css('div.docsum-content'):
        yield {
            'title': item.css('a.docsum-title::text').getall(),
            'discription': item.css('div.full-view-snippet::text').getall()
         }
