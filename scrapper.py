import feedparser as fp
import dateutil.parser
from newspaper import Article, Config
import logging
import pandas as pd
import json
from datetime import datetime, timedelta, timezone
from textblob import TextBlob
import os
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
from nltk.tokenize import word_tokenize
from unidecode import unidecode

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CacheManager:
    def __init__(self, cache_file='article_cache.json'):
        self.cache_file = cache_file
        self.load_cache()
    
    def load_cache(self):
        logging.info("Loading cache")
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
    
    def save_cache(self):
        logging.info("Saving cache")
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=4)
    
    def get_article(self, link):
        return self.cache.get(link)
    
    def set_article(self, link, article):
        self.cache[link] = article

class NewsScraper:
    def __init__(self, sources_file='app/sources.json', user_agent='Mozilla/5.0'):
        self.sources_file = sources_file
        self.user_agent = user_agent
        self.article_list = []
        self.config = Config()
        self.config.browser_user_agent = self.user_agent
        self.cache_manager = CacheManager()

    def parse_feed(self, source_name, feed_url):
        feed = fp.parse(feed_url)
        articles = []

        for entry in feed.entries:
            if 'published' in entry:
                published = dateutil.parser.parse(entry.published)
            elif 'updated' in entry:
                published = dateutil.parser.parse(entry.updated)
            else:
                continue
            
            article = {
                'source': source_name,
                'title': entry.title,
                'link': entry.link,
                'published': published,
            }
            articles.append(article)

        return articles

    def fetch_article_content(self, article):
        if (cached_article := self.cache_manager.get_article(article['link'])):
            return cached_article

        url = article['link']
        news_article = Article(url, config=self.config)

        try:
            news_article.download()
            news_article.parse()
            news_article.nlp()
        except Exception as e:
            logging.warning(f"Failed to download or parse article from {url}: {e}")
            return None

        article_data = {
            'url': url,
            'title': news_article.title,
            'body': news_article.text,
            'keywords': news_article.keywords,
            'summary': news_article.summary,
            'image_url': news_article.top_image,
            'date': article['published'].strftime('%Y-%m-%d'),
            'time': article['published'].strftime('%H:%M:%S'),
        }

        sentiment = TextBlob(article_data['body']).sentiment
        if sentiment.polarity > 0.1:
            sentiment_category = 'positive'
        elif sentiment.polarity < -0.1:
            sentiment_category = 'negative'
        else:
            sentiment_category = 'neutral'
        
        article_data['sentiment'] = sentiment.polarity
        article_data['sentiment_category'] = sentiment_category

        self.cache_manager.set_article(url, article_data)
        return article_data

    def scrape_sources(self):
        with open(self.sources_file, 'r') as f:
            sources = json.load(f)

        for source in sources:
            logging.info(f"Scraping articles from {source['name']}")
            feed_articles = self.parse_feed(source['name'], source['feed_url'])
            for article in feed_articles:
                article_content = self.fetch_article_content(article)
                if article_content:
                    self.article_list.append(article_content)

        self.cache_manager.save_cache()
        return self.article_list

def main():
    scraper = NewsScraper()
    articles = scraper.scrape_sources()
    logging.info(f"Scraped {len(articles)} articles")

if __name__ == "__main__":
    main()
