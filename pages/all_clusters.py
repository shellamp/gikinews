import pandas as pd
import streamlit as st
import json
import toml
from clustering import compute_tfidf
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

# PAGE FORMAT
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Load configuration from TOML file
config = toml.load('config.toml')

# Header for the Streamlit app
st.title('Giki News for People in a Hurry!')

# Load the JSON file with article data
file_path = 'article_cache.json'

@st.cache_data
def load_data():
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load the data from the JSON file
original_data = load_data()

# Extract necessary data from the loaded JSON
articles = list(original_data.values())

# Sidebar filters
st.sidebar.header('Filters')
search_topic = st.sidebar.text_input("Search for a topic")
selected_sentiment = st.sidebar.multiselect(
    "Select Sentiment Category",
    options=list(set(article['sentiment_category'] for article in articles)),
    default=list(set(article['sentiment_category'] for article in articles))
)
all_sources = list(set(article['source'] for article in articles))
selected_sources = st.sidebar.multiselect(
    "Select Sources",
    options=all_sources,
    default=[]
)

# If no sources are selected, use all sources
if not selected_sources:
    selected_sources = all_sources

# Filter articles based on the search topic, selected sentiment category, and selected sources
filtered_articles = [
    article for article in articles 
    if (search_topic.lower() in article['title'].lower() or search_topic.lower() in article['body'].lower())
    and article['sentiment_category'] in selected_sentiment
    and article['source'] in selected_sources
]

# Determine clusters using AgglomerativeClustering
if len(filtered_articles) == 0:
    st.write("No articles found")
else:
    # Compute TF-IDF values for filtered articles
    news_df = pd.DataFrame(filtered_articles)
    tfidf_array = compute_tfidf(news_df)
    
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
    news_df['cluster_id'] = clustering_model.fit_predict(tfidf_array)
    
    clusters = {str(cluster_id): news_df[news_df.cluster_id == cluster_id].to_dict(orient='records')
                for cluster_id in news_df['cluster_id'].unique()}
    
    # Calculate most frequent keywords for each cluster
    cluster_keywords = {}
    for cluster_id, articles in clusters.items():
        keywords = [keyword for article in articles for keyword in article.get('keywords', [])]
        most_common_keywords = [keyword for keyword, _ in Counter(keywords).most_common(3)]
        cluster_keywords[cluster_id] = most_common_keywords

    # Display clusters with articles
    st.header("Article Clusters")

    # Sort clusters by cluster number
    sorted_clusters = sorted(clusters.items())

    for cluster_id, articles in sorted_clusters:
        # Display cluster number and sample keywords
        st.subheader(f'Cluster {cluster_id}')
        cluster_keywords_list = ", ".join(cluster_keywords[cluster_id])
        st.write(f'**Keywords:** {cluster_keywords_list}')
        
        # Display articles in a three-column layout
        cols = st.columns(3)
        for idx, article in enumerate(articles[:3]):
            with cols[idx]:
                image_url = article.get('image_url', 'https://via.placeholder.com/150')
                date = article.get('date', 'No date available')
                title = article.get('title', 'No title available')
                body = article.get('body', 'No body available')
                sentiment = article.get('sentiment_category', 'No sentiment category available')
                url = article.get('url', '#')

                # Truncate body to 100 words
                truncated_body = " ".join(body.split()[:100]) + '...' if len(body.split()) > 100 else body

                # Display article details
                st.image(image_url, use_column_width=True)
                st.write(f"Source: {article['source']}")
                st.write(f"Published on: {date}")
                st.markdown(f"[**{title}**]({url})")
                st.write(truncated_body)
                st.write(f"Sentiment: {sentiment}")
