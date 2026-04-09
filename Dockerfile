FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r /app/requirements.txt

# Copy the API code
COPY app.py /app/app.py

# Copy the promoted champion model artifact from the registry location.
COPY modelinfo/modelregistry/champion/model.joblib /app/modelinfo/modelregistry/champion/model.joblib

# Ensure the service loads the registry champion by default.
ENV MODEL_PATH=/app/modelinfo/modelregistry/champion/model.joblib

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
