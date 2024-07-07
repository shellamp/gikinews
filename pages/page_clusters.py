# import pandas as pd
# import os
# import json
# import numpy as np
# from datetime import datetime, timedelta
# from sklearn.cluster import AgglomerativeClustering
# from pathlib import Path
# import base64
# from collections import Counter
# import altair as alt

import pandas as pd
import streamlit as st
import json
import toml
from clustering import cluster_articles
from collections import Counter

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

# Check if cluster ID is provided
cluster_id = int(st.query_params.get('cluster_id', [0])[0])

# Load data from the JSON file
original_data = load_data()

# Extract necessary data from the loaded JSON
articles = list(original_data.values())

# Re-cluster the articles
filtered_titles = [article['title'] for article in articles]

# Determine the number of clusters dynamically
filtered_total_articles = len(filtered_titles)
if filtered_total_articles == 0:
    st.write("No articles found")
elif 0 < filtered_total_articles <= 10:
    num_clusters = 3
elif 10 < filtered_total_articles <= 50:
    num_clusters = 5
elif 50 < filtered_total_articles <= 100:
    num_clusters = 7
else:
    num_clusters = 12

# Cluster articles based on titles
clusters = cluster_articles(filtered_titles, num_clusters)

# Get articles in the selected cluster
if cluster_id in clusters:
    cluster_articles = [article for article in articles if article['title'] in clusters[cluster_id]]

    # Display articles in the cluster
    st.title(f"Articles in Cluster {cluster_id + 1}")

    for article in cluster_articles:
        st.markdown(f"## {article['title']}")
        st.image(article.get('image_url', ''), use_column_width=True)
        st.markdown(f"**Source:** {article.get('source', 'N/A')}")
        st.markdown(f"**Published on:** {article.get('published_date', 'N/A')}")
        st.markdown(article['body'])
        st.markdown(f"**Frequent Words:** {', '.join(article.get('keywords', []))}")
        st.markdown(f"**Sentiment:** {article.get('sentiment_category', 'N/A')}")
        st.markdown("---")
else:
    st.write(f"No articles found for cluster {cluster_id + 1}.")

# @st.cache_data(ttl=3600)  
# def load_articles_from_cache(cache_file):
#     try:
#         if os.path.exists(cache_file):
#             with open(cache_file, 'r') as f:
#                 articles = json.load(f)
#             return pd.DataFrame(articles.values())
#         else:
#             return pd.DataFrame()
#     except Exception as e:
#         st.error(f"Error loading cache: {e}")
#         return pd.DataFrame()

# def filter_articles_by_keywords(articles, keywords):
#     if not isinstance(keywords, list):
#         keywords = [keywords] if keywords else []

#     # Ensure keywords are not empty or None
#     keywords = [keyword.lower() for keyword in keywords if keyword]

#     if not keywords:
#         return articles  # Return all articles if no valid keywords are provided

#     filtered_articles = []
#     for article in articles:
#         body = article.get('body', '').lower()  # Convert body to lowercase for case-insensitive matching
#         if any(keyword in body for keyword in keywords):
#             filtered_articles.append(article)

#     return filtered_articles

# def filter_articles_by_date_and_sentiment(articles_df, start_date, end_date, sentiment):
#     if 'date' in articles_df.columns:
#         articles_df['date'] = pd.to_datetime(articles_df['date'])
#         start_date = pd.to_datetime(start_date)
#         end_date = pd.to_datetime(end_date)
#         articles_df = articles_df[(articles_df['date'] >= start_date) & (articles_df['date'] <= end_date)]
    
#     if sentiment and 'sentiment_category' in articles_df.columns:
#         articles_df = articles_df[articles_df['sentiment_category'] == sentiment]

#     return articles_df

# def cluster_articles(articles_df, keyword):
#     if 'body' not in articles_df.columns:
#         st.error("Missing 'body' column in the articles data.")
#         return pd.DataFrame(), []

#     articles_df['body'] = articles_df['body'].astype(str).fillna('')

#     if keyword:
#         articles_df = articles_df[articles_df['body'].str.contains(keyword, case=False, na=False)]

#     if articles_df.empty:
#         return pd.DataFrame(), []

#     articles_df.fillna('', inplace=True)

#     tfidf_df = compute_tfidf(articles_df)
#     distance_threshold = 1.5
#     ac = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None)
#     articles_labeled = ac.fit_predict(tfidf_df)

#     articles_df['cluster_id'] = articles_labeled
#     clusters = {str(n): articles_df[articles_df['cluster_id'] == n].to_dict(orient='records') for n in np.unique(articles_labeled)}

#     return articles_df, clusters

# def truncate_summary(summary, word_limit=100):
#     words = summary.split()
#     if len(words) > word_limit:
#         return ' '.join(words[:word_limit]) + '...'
#     return summary

# def display_article(article):
#     if article.get('image_url'):
#         st.image(article['image_url'], use_column_width=True)

#     st.markdown(f"### [{article.get('title')}]({article.get('url')})")
#     st.subheader(f"Source: {article.get('source')}")
#     st.write(f"Published on: {article.get('date')} at {article.get('time')}")

#     summary = article.get('summary', '')
#     truncated_summary = truncate_summary(summary)
#     st.write(truncated_summary)

#     st.write(f"Frequent Words: {', '.join(article.get('keywords', []))}")
#     st.write(f"Sentiment: {article.get('sentiment_category')}")
#     st.write(f"Cluster ID: {article.get('cluster_id')}")
#     st.write("---")

# def display_articles(articles_df, clusters, clusters_per_row=3):
#     if articles_df.empty:
#         st.write("No articles found with the given keyword or current date.")
#         return

#     grouped = articles_df.groupby('cluster_id')
#     cluster_ids = sorted(grouped.groups.keys())

#     for i in range(0, len(cluster_ids), clusters_per_row):
#         cluster_subset = cluster_ids[i:i + clusters_per_row]
#         cols = st.columns(len(cluster_subset))

#         for col, cluster_id in zip(cols, cluster_subset):
#             with col:
#                 st.markdown(f"## Cluster {cluster_id}")
#                 group = grouped.get_group(cluster_id)
#                 articles = group.iterrows()
#                 displayed_articles = 0

#                 for _, article in articles:
#                     if displayed_articles < 2:
#                         display_article(article)
#                         displayed_articles += 1
#                     else:
#                         break

#                 if displayed_articles < len(group):
#                     with st.expander("Show more articles"):
#                         for _, article in articles:
#                             display_article(article)

# def extract_frequent_terms_from_keywords(articles, top_n=20):
#     all_keywords = []
#     for article in articles:
#         keywords = article.get('keywords', [])
#         all_keywords.extend(keywords)
    
#     keyword_freq = Counter(all_keywords)
#     common_terms = keyword_freq.most_common(top_n)
#     return common_terms

# def plot_frequent_terms(terms):
#     df = pd.DataFrame(terms, columns=['term', 'frequency'])
#     df['size'] = df['frequency'] * 10

#     chart = alt.Chart(df).mark_circle().encode(
#         x=alt.X('term:N', sort=alt.EncodingSortField(field='frequency', order='descending')),
#         y='frequency:Q',
#         size='size:Q',
#         color='term:N',
#         tooltip=['term', 'frequency']
#     ).properties(
#         title='Most Frequent Terms',
#         width=800,
#         height=400
#     )

#     st.altair_chart(chart)

# def img_to_bytes(img_path):
#     img_bytes = Path(img_path).read_bytes()
#     encoded = base64.b64encode(img_bytes).decode()
#     return encoded

# def img_to_html(img_path):
#     img_html = "<img src='data:image/png;base64,{}' style='width:100%;' class='img-fluid'>".format(
#       img_to_bytes(img_path)
#     )
#     return img_html

# if __name__ == '__main__':

#     st.title("News Articles for people on a hurry!")

#     # Sidebar filters
#     with st.sidebar:
#         # Display keyword logo above the search bar
#         st.markdown(
#             f"""
#             <div style="width: 355px;display: flex; justify-content: center; align-items: center;margin-left:-40px">
#                 {img_to_html(KEYWORD_LOGO_PATH)}
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
        
#         keyword = st.text_input("Search articles by keyword")

#         articles_df = load_articles_from_cache(ARTICLES_CACHE_FILE)
        
#         if not articles_df.empty:
#             articles_df['date'] = pd.to_datetime(articles_df['date'])
#             min_date = articles_df['date'].min().date()
#             max_date = articles_df['date'].max().date()
#         else:
#             st.error("No articles found in cache.")
#             min_date = datetime.today().date() - timedelta(days=30)
#             max_date = datetime.today().date()

#         # Date filter slider with dynamic date range
#         start_date, end_date = st.slider(
#             "Filter articles by publication date",
#             min_value=min_date,
#             max_value=max_date,
#             value=(max_date, max_date) if not keyword else (min_date, max_date),
#             format="YYYY-MM-DD"
#         )

#         # Sentiment filter dropdown
#         sentiment = st.selectbox(
#             "Filter articles by sentiment",
#             options=["", "negative", "neutral", "positive"],
#             format_func=lambda x: "All" if x == "" else x.capitalize()
#         )

#         st.markdown(
#         f"""
#         <div style="display: flex; justify-content: center; align-items: center; background-color: #2E3859; border-radius: 10px;width:500px;margin:0 auto;margin-top:-55px;">
#             {img_to_html(LOGO_PATH)}
#         </div>
#         """,
#         unsafe_allow_html=True
#          )

#     if keyword:
#         filtered_articles = filter_articles_by_keywords(articles_df.to_dict(orient='records'), [keyword])
#         filtered_articles_df = pd.DataFrame(filtered_articles)
#     else:
#         filtered_articles_df = articles_df

#     filtered_articles_df = filter_articles_by_date_and_sentiment(filtered_articles_df, start_date, end_date, sentiment)
    
#     # Display metrics in a column layout
#     col1, col2 = st.columns(2)
#     total_articles_scraped = len(articles_df)
#     total_articles_filtered = len(filtered_articles_df)
#     col1.metric("Total Articles Scraped", total_articles_scraped)
#     col2.metric("Total Articles Based on Filter", total_articles_filtered)

#     if len(filtered_articles_df) == 1:
#         # Display the single article directly without clustering
#         st.markdown("## Single Article Found")
#         display_article(filtered_articles_df.iloc[0])
#     else:
#         filtered_articles_df, clusters = cluster_articles(filtered_articles_df, keyword)
#         display_articles(filtered_articles_df, clusters)
    
#     # Extract and plot frequent terms
#     if not filtered_articles_df.empty:
#         terms = extract_frequent_terms_from_keywords(filtered_articles_df.to_dict(orient='records'))
#         if terms:
#             st.markdown("## Most Frequent Terms in Filtered Articles")
#             plot_frequent_terms(terms)
