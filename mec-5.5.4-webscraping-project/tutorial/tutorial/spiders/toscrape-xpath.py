import scrapy

class ToScrapeXPATHpider(scrapy.Spider):
    name = "toscrape-xpath"

    start_urls = [
        'http://quotes.toscrape.com/',
        
        ]

    def parse(self, response):
        for quote in response.xpath("//div[@class and contains(concat(' ', normalize-space(@class), ' '), ' quote ')]"):
            yield {
                'text': quote.xpath("//span[@class and contains(concat(' ', normalize-space(@class), ' '), ' text ')]/text()").extract_first(),
                'author': quote.xpath("//small[@class and contains(concat(' ', normalize-space(@class), ' '), ' author ')]/text()").extract_first(),
                'tags': quote.xpath("//div[@class and contains(concat(' ', normalize-space(@class), ' '), ' tags ')]/descendant-or-self::*/a[@class and contains(concat(' ', normalize-space(@class), ' '), ' tag ')]/text()").extract(),
            }

        next_page_url = response.xpath("//li[@class and contains(concat(' ', normalize-space(@class), ' '), ' next ')]/a/@href").extract_first()
        
        if next_page_url is not None:                
            yield scrapy.Request(response.urljoin(next_page_url))






