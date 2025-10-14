

%%writefile src/crawler.py
import os
import time
import json
import requests
from urllib.parse import urljoin, urlparse
from collections import deque
from typing import List, Dict, Set, Tuple
import trafilatura
from robotexclusionrulesparser import RobotExclusionRulesParser

class WebCrawler:
    def __init__(self, start_url: str, max_pages: int = 50, 
                 max_depth: int = 3, crawl_delay_ms: int = 500):
        self.start_url = start_url
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.crawl_delay_ms = crawl_delay_ms
        self.visited_urls: Set[str] = set()
        self.documents: List[Dict] = []
        self.skipped_count = 0
      
        
        parsed = urlparse(start_url)
        self.domain = parsed.netloc
        self.base_domain = '.'.join(parsed.netloc.split('.')[-2:])
        
        # Setup robots.txt parser
        self.robots_parser = RobotExclusionRulesParser()
        try:
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            robots_content = requests.get(robots_url, timeout=10).text
            self.robots_parser.parse(robots_content)
        except:
            pass  
    
    def is_same_domain(self, url: str) -> bool:
       
        parsed = urlparse(url)
        url_domain = '.'.join(parsed.netloc.split('.')[-2:])
        return url_domain == self.base_domain
    
    def can_fetch(self, url: str) -> bool:
        """Check robots.txt compliance"""
        return self.robots_parser.is_allowed("*", url)
    
    def extract_text(self, url: str) -> Tuple[str, str]:
        
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'RAGBot/1.0 (Educational Project)'
            })
            response.raise_for_status()
            
          
            text = trafilatura.extract(response.content, 
                                      include_comments=False,
                                      include_tables=True)
            
            if text and len(text.strip()) > 100:
                return text.strip(), response.text
            return None, None
            
        except Exception as e:
            print(f"Error extracting {url}: {str(e)}")
            return None, None
    
    def get_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML"""
        from bs4 import BeautifulSoup
        links = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                
               
                parsed = urlparse(full_url)
                normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                
                if self.is_same_domain(normalized) and normalized not in self.visited_urls:
                    links.append(normalized)
        except:
            pass
        return links
    
    def crawl(self) -> Dict:
       
        queue = deque([(self.start_url, 0)]) 
        
        print(f"Starting crawl from {self.start_url}")
        print(f"Max pages: {self.max_pages}, Max depth: {self.max_depth}")
        
        while queue and len(self.documents) < self.max_pages:
            url, depth = queue.popleft()
            
            if url in self.visited_urls or depth > self.max_depth:
                continue
            
            if not self.can_fetch(url):
                self.skipped_count += 1
                continue
            
            self.visited_urls.add(url)
            
            # Extract content
            text, html = self.extract_text(url)
            
            if text:
                self.documents.append({
                    'url': url,
                    'text': text,
                    'depth': depth,
                    'length': len(text)
                })
                print(f"[{len(self.documents)}/{self.max_pages}] Crawled: {url[:80]}")
                
               
                if depth < self.max_depth:
                    links = self.get_links(html, url)
                    for link in links[:10]:  
                        queue.append((link, depth + 1))
            else:
                self.skipped_count += 1
            
          
            time.sleep(self.crawl_delay_ms / 1000.0)
        
        # Save documents
        with open('data/crawled_documents.json', 'w') as f:
            json.dump(self.documents, f, indent=2)
        
        return {
            'page_count': len(self.documents),
            'skipped_count': self.skipped_count,
            'urls': [doc['url'] for doc in self.documents]
        }
