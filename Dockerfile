
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

# Creates directory for data if it doesn't exist (though ideally data should be mounted)
RUN mkdir -p /app/data

# Expose API port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
