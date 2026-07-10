FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose both FastAPI and Streamlit ports
EXPOSE 8000 8501

# Command to run both services using a shell script or we can rely on docker-compose for splitting them
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
