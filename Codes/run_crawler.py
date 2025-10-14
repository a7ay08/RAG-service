
from src.crawler import WebCrawler

# Initialize the crawler
crawler = WebCrawler(
    start_url="https://en.wikipedia.org/wiki/Python_(programming_language)",
    max_pages=30,
    max_depth=4,
    crawl_delay_ms=500
)

# Start crawling
crawl_result = crawler.crawl()
print(f"\n✅ Crawl complete!")
print(f"Pages crawled: {crawl_result['page_count']}")
print(f"Pages skipped: {crawl_result['skipped_count']}")
