import pandas as pd
import streamlit as st
import json
import toml
from clustering import compute_tfidf
from sklearn.cluster import AgglomerativeClustering

# PAGE FORMAT
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Load configuration from TOML file
config = toml.load('config.toml')

# Load the JSON file with article data
file_path = 'article_cache.json'

@st.cache_data
def load_data():
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load configuration from TOML file
config = toml.load('config.toml')

# Define custom CSS for the Streamlit app
st.markdown(f"""
    <style>
        body {{
            color: {config['theme']['textColor']};
            background-color: {config['theme']['backgroundColor']};
            font-family: {config['theme']['font']};
            font-size: {config['theme']['fontSize']}px;
        }}
        .primary {{
            color: {config['theme']['primaryColor']};
        }}
        .secondary-bg {{
            background-color: {config['theme']['secondaryBackgroundColor']};
        }}
    </style>
    """, unsafe_allow_html=True)

# Sidebar filters
st.sidebar.image("app/logo.png", use_column_width=True)

# Extract cluster ID from URL query parameters
query_params = st.experimental_get_query_params()
saved_cluster_id = int(query_params.get('cluster_id', [0])[0])

# Load data from the JSON file
original_data = load_data()

# Extract necessary data from the loaded JSON
articles = list(original_data.values())

# Extract titles and bodies for clustering
titles = [article['title'] for article in articles]
bodies = [article['body'] for article in articles]

# Compute TF-IDF values for filtered articles
news_df = pd.DataFrame(original_data).T
tfidf_array = compute_tfidf(news_df)

clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
news_df['cluster_id'] = clustering_model.fit_predict(tfidf_array)

# Organize articles by cluster
clusters = {cluster: news_df[news_df.cluster_id == cluster]['title'].tolist()
            for cluster in news_df['cluster_id'].unique()}

def truncate_text(text, max_words=100):
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return text

# Get articles in the selected cluster
if saved_cluster_id in clusters:
    cluster_articles = [article for article in articles if article['title'] in clusters[saved_cluster_id]]

    # Display articles in the cluster
    st.title(f"Articles in Cluster {saved_cluster_id}")

    cols = st.columns(3)  # Create 3 columns for displaying articles
    for idx, article in enumerate(cluster_articles):
        col = cols[idx % 3]  # Select column for the current article
        with col:
            st.markdown(f"## {article['title']}")
            st.image(article.get('image_url', ''), use_column_width=True)
            st.markdown(f"**Source:** {article.get('source', 'N/A')}")
            st.markdown(f"**Published on:** {article.get('date', 'N/A')}")
            st.markdown(truncate_text(article['body']))
            st.markdown(f"**Frequent Words:** {', '.join(article.get('keywords', []))}")
            st.markdown(f"**Sentiment:** {article.get('sentiment_category', 'N/A')}")
            st.markdown("---")
else:
    st.write(f"No articles found for cluster {saved_cluster_id}.")
