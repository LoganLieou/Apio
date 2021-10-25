import scrapy

class MedSpider(scrapy.Spider):
   name = 'medspider'
   start_urls = ["https://pubmed.ncbi.nlm.nih.gov/?term=deep%20learning&page=" + str(x) for x in range(1, 3026)]

   def parse(self, response):
      for item in response.css('a.docsum-title'):
         yield {
            'title': item.css('::text').get()
         }
