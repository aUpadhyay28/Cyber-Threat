FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY riskengine ./riskengine
COPY scripts ./scripts
COPY models ./models
ENV RISK_MODEL_PATH=models/latest.joblib RISK_STATE_PATH=/data/state.pkl RISK_DB_PATH=/data/risk.db
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "riskengine.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
