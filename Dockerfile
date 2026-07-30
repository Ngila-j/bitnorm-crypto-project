# BitNorm BN Analytics – Streamlit + API image (Phase 4)
FROM python:3.11-slim
 
WORKDIR /app
 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BN_ENV=production \
    EXCHANGE_MODE=mock
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY . .
 
# Generate demo data on first build layer (optional; also runs at runtime if missing)
RUN python pipeline.py || true
 
EXPOSE 8501 8000
 
# Default: Streamlit terminal. Override CMD for API-only.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]