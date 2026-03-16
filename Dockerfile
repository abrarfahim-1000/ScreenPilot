FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better caching)
COPY gcrequirements.txt .

# Clean install — strips any non-pip lines from gcrequirements.txt
RUN grep -E '^[a-zA-Z]' gcrequirements.txt > clean_requirements.txt && \
    pip install --no-cache-dir -r clean_requirements.txt

# Copy server files directly into /app
COPY server/app.py .
COPY server/schemas.py .
COPY server/gemini.py .
COPY server/firestore_session.py .
COPY server/gcs_storage.py .

# Explicitly set port
EXPOSE 8080
ENV PORT=8080

# Use explicit port number, not shell variable expansion
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]