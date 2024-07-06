#!/bin/sh

# Run the scrapper script
python scrapper.py

# Run the clustering script
python clustering.py

# Start the Streamlit app
streamlit run app.py
