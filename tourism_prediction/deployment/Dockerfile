# Use Python 3.9 base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy dependency list first (better caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . /app

# Run Streamlit on Hugging Face default port (7860)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
