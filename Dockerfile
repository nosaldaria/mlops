FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.10-slim
WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY config/ ./config/
COPY dags/ ./dags/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

CMD ["python", "src/optimize.py", "--help"]