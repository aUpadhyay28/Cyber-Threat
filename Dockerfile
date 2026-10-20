FROM python:3.11-slim

# Run as a normal user: a container that only scores logins has no reason to be root, and the risk
# service holds the credential-stuffing history for every account on the site.
RUN useradd --create-home --uid 10001 risk
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY riskengine ./riskengine
COPY scripts ./scripts
# The model is NOT baked in. models/ is gitignored, so `COPY models ./models` quietly produced an
# image whose only content was .gitkeep and which died on startup. Mount it instead:
#   docker run -v $PWD/models:/app/models:ro -v risk-data:/data ...
# and pin it with scripts/pin_model.py so a swapped file cannot be loaded.

ENV RISK_MODEL_PATH=/app/models/latest.joblib \
    RISK_STATE_PATH=/data/state.json.gz \
    RISK_DB_PATH=/data/risk.db \
    RISK_TIMESTAMP_SOURCE=server \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data /app/models && chown -R risk:risk /data /app
VOLUME /data
USER risk
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

CMD ["uvicorn", "riskengine.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
