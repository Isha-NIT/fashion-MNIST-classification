# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy frontend requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --timeout 1000 --retries 20 -i https://pypi.org/simple -r requirements.txt

# Copy all frontend files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
