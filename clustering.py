import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import json
import logging

CACHE_FILE = 'article_cache.json'

class Helper:
    @staticmethod
    def print_scrape_status(count):
        logging.info(f'Scraped {count} articles so far...')

    @staticmethod
    def clean_dataframe(news_df):
        logging.info("Cleaning dataframe")
        news_df = news_df[news_df.title != '']
        news_df = news_df[news_df.body != '']
        news_df = news_df[news_df.image_url != '']
        news_df = news_df[news_df.title.str.count(r'\s+').ge(3)]
        news_df = news_df[news_df.body.str.count(r'\s+').ge(20)]
        return news_df

def compute_tfidf(news_df):
    logging.info("Computing TF-IDF values")
    tfidf_matrix = TfidfVectorizer().fit_transform(news_df['clean_body'])
    tfidf_array = np.asarray(tfidf_matrix.todense())
    return tfidf_array

def find_featured_clusters(clusters):
    logging.info("Finding clusters with articles from multiple sources")
    featured_clusters = {}
    for i in clusters.keys():
        if len(set([j["source"] for j in clusters[i]])) > 1:
            featured_clusters[i] = clusters[i]
    return featured_clusters

def main():
    logging.info("Loading articles from cache")
    with open(CACHE_FILE, 'r') as f:
        articles = json.load(f)
    
    news_df = pd.DataFrame(articles.values())
    helper = Helper()
    news_df = helper.clean_dataframe(news_df)
    tfidf_array = compute_tfidf(news_df)
    
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
    news_df['cluster_id'] = clustering_model.fit_predict(tfidf_array)
    
    clusters = {str(cluster_id): news_df[news_df.cluster_id == cluster_id].to_dict(orient='records')
                for cluster_id in np.unique(news_df.cluster_id)}
    
    featured_clusters = find_featured_clusters(clusters)
    
    logging.info("Saving clusters to cache")
    with open(CACHE_FILE, 'w') as f:
        json.dump(articles, f, indent=4)

if __name__ == "__main__":
    main()

def cluster_articles(news_df):
    num_articles = len(news_df)
    
    if num_articles < 3:
        return {}, news_df
    
    if num_articles < 5:
        num_clusters = 2
    elif num_articles < 20:
        num_clusters = 3
    elif num_articles < 50:
        num_clusters = 6
    else:
        num_clusters = 10
    
    tfidf_array = compute_tfidf(news_df)
    clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
    cluster_labels = clustering_model.fit_predict(tfidf_array)
    
    news_df['cluster_id'] = cluster_labels
    clusters = {}
    
    for idx, row in news_df.iterrows():
        cluster_id = row['cluster_id']
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(row.to_dict())
    
    return clusters, news_df
