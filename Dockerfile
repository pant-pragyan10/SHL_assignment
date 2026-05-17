FROM python:3.11-slim

RUN useradd --create-home appuser || true
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN chmod +x /app/entrypoint.sh || true

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

USER appuser
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "src.shl_agent.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "30"]
