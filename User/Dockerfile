FROM python:3.11

EXPOSE 8084
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the entrypoint to run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8084"]
