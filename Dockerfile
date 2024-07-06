# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Ensure the images are in the correct path inside the container
COPY app/Cat.jpg app/Cat.jpg
COPY app/logo.jpg app/logo.jpg

# Make entrypoint.sh executable
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Use entrypoint.sh as the entrypoint for the container
ENTRYPOINT ["./entrypoint.sh"]
CMD ["streamlit", "run", "--server.enableCORS", "false", "--server.port", "8503", "main.py"]
