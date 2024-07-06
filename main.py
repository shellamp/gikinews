import pandas as pd
import streamlit as st
import altair as alt
import json
import toml
import os
from clustering import cluster_articles

# Other necessary imports...

# PAGE FORMAT
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

config = toml.load('config.toml')

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

st.markdown(f'<h1 class="primary">Giki News for people in a hurry!</h1>', unsafe_allow_html=True)

# DATA LOADING
@st.cache_data
def load_data():
    file_path = 'article_cache.json'
    with open(file_path, 'r') as f:
        data = json.load(f)

    reformatted_data = []
    for url, content in data.items():
        entry = {
            "source": content.get("source", ""),
            "url": content.get("url", ""),
            "date": content.get("date", ""),
            "time": content.get("time", ""),
            "title": content.get("title", ""),
            "body": content.get("body", ""),
            "summary": content.get("summary", ""),
            "keywords": content.get("keywords", []),
            "image_url": content.get("image_url", ""),
            "clean_body": content.get("clean_body", ""),
            "cluster_id": content.get("cluster_id", None),
            "sentiment": content.get("sentiment", ""),
            "sentiment_category": content.get("sentiment_category", ""),
            "contains_keyword": content.get("contains_keyword", False)
        }
        reformatted_data.append(entry)

    df = pd.DataFrame(reformatted_data)
    return df

papers_df = load_data()
papers_df = papers_df.explode('keywords')

required_columns = ["source", "title", "keywords", "date", "body", "sentiment_category"]
missing_columns = [col for col in required_columns if col not in papers_df.columns]

if missing_columns:
    st.error(f"Missing columns in the data: {', '.join(missing_columns)}")

# KEYWORD PREPROCESSING
min_count = 8

papers_df['keywords'] = papers_df['keywords'].explode().str.strip().str.lower()

keyword_counts = papers_df['keywords'].value_counts()
frequent_keywords = keyword_counts[keyword_counts >= min_count].index.tolist()
papers_df = papers_df[papers_df['keywords'].isin(frequent_keywords)]

# DEFINE VARIABLES
sources = sorted(papers_df["source"].dropna().unique())
titles = sorted(papers_df["title"].dropna().unique())
keywords = sorted(papers_df["keywords"].dropna().unique())
dates = sorted(papers_df["date"].dropna().unique())
sentiment_categories = sorted(papers_df["sentiment_category"].dropna().unique())

# SIDE BAR CONFIGURATION
st.sidebar.image("app/Cat.jpg", use_column_width=True)
st.sidebar.header('Select topics')

search_topic = st.text_input("Search articles by topic")

st.sidebar.subheader('Bubble Chart')
bubble_chart_param = st.sidebar.selectbox('Select parameter', ('keywords', 'source', 'sentiment_category'))

st.sidebar.subheader('Donut Chart')
donut_chart_param = st.sidebar.selectbox('Select parameter', ('source', 'keywords', 'sentiment_category'))

st.sidebar.subheader('Sentiment category filter')
selected_sentiment_categories = st.sidebar.multiselect("Select parameter", sentiment_categories)

st.sidebar.markdown('''
---
Giki News
''')

# SELECTBOXES
selected_sources = st.multiselect("Select Sources", sources)

# Apply filters to DataFrame based on selected filters
filtered_papers_df = papers_df

if selected_sources:
    filtered_papers_df = filtered_papers_df[filtered_papers_df["source"].isin(selected_sources)]

# if selected_keywords:
#     filtered_papers_df = filtered_papers_df[filtered_papers_df["keywords"].isin(selected_keywords)]

if selected_sentiment_categories:
    filtered_papers_df = filtered_papers_df[filtered_papers_df["sentiment_category"].isin(selected_sentiment_categories)]

if search_topic:
    search_topic = search_topic.lower()
    filtered_papers_df = filtered_papers_df[
        filtered_papers_df["title"].str.lower().str.contains(search_topic) |
        filtered_papers_df["body"].str.lower().str.contains(search_topic)
    ]

# Clustering
clusters, clustered_papers_df = cluster_articles(filtered_papers_df)

# KPI VISUALS
source_count = str(filtered_papers_df["source"].nunique())
keyword_count = str(filtered_papers_df["keywords"].nunique())
article_count = str(filtered_papers_df["title"].nunique())

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Total Articles", value=article_count)
with col2:
    st.metric(label="Total Sources", value=source_count)
with col3:
    st.metric(label="Total Keywords", value=keyword_count)

# CHARTS
col3, col4 = st.columns((7, 3))

with col3:
    st.subheader("Bubble Chart")
    bubble_chart_data = filtered_papers_df.groupby(bubble_chart_param).size().reset_index(name='counts')
    bubble_chart = alt.Chart(bubble_chart_data).mark_circle().encode(
        x=alt.X(bubble_chart_param, title=bubble_chart_param.capitalize()),
        y=alt.Y('counts', title='Number of Articles'),
        size='counts',
        color=bubble_chart_param,
        tooltip=[bubble_chart_param, 'counts']
    ).properties(
        width=600,
        height=300
    )
    st.altair_chart(bubble_chart, use_container_width=True)

with col4:
    st.subheader("Donut Chart")
    donut_chart_data = filtered_papers_df[donut_chart_param].value_counts().reset_index()
    donut_chart_data.columns = [donut_chart_param, 'counts']
    donut_chart = alt.Chart(donut_chart_data).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field='counts', type='quantitative'),
        color=alt.Color(field=donut_chart_param, type='nominal'),
        tooltip=[donut_chart_param, 'counts']
    ).properties(
        width=100,
        height=200
    )
    st.altair_chart(donut_chart, use_container_width=True)

# Display clustering results
num_clusters = len(clusters)
st.subheader(f"Number of clusters created: {num_clusters}")

if num_clusters > 0:
    for cluster_id, articles in clusters.items():
        st.write(f"### Cluster {cluster_id}")
        st.write(f"**Number of articles in cluster**: {len(articles)}")
        st.write("**Short explanation**:")
        st.write(f"Articles in this cluster are about '{', '.join([article['title'] for article in articles[:3]])}...'")

    # Display articles in a table, where columns are clusters
    cluster_table = pd.DataFrame.from_records([{'title': article['title'], 'cluster': article['cluster_id']} for article in clustered_papers_df.to_dict('records')])
    cluster_table_pivot = cluster_table.pivot(columns='cluster', values='title')
    st.dataframe(cluster_table_pivot)