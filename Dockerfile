# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install lxml[html_clean]
RUN pip install lxml_html_clean

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Ensure the images are in the correct path inside the container
COPY app/Cat.png app/Cat.png
COPY app/logo.png app/logo.png

# Make entrypoint.sh executable
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Use entrypoint.sh as the entrypoint for the container
ENTRYPOINT ["./entrypoint.sh"]
CMD ["streamlit", "run", "--server.enableCORS", "false", "--server.port", "8501", "main_page.py"]
