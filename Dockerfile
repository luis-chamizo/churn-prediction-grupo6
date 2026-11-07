# Imagen para desplegar la aplicación en cualquier servicio con Docker
# (Render, Railway, Hugging Face Spaces, Google Cloud Run, etc.)
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["sh", "-c", "streamlit run app/Inicio.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
