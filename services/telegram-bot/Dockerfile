# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install runtime deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app

ENV TELEGRAM_BOT_TOKEN="" \
    BOT_NAME="RADQ Expenses Tracker" \
    BOT_SHORT_DESCRIPTION="Track expenses easily with RADQ." \
    BOT_DESCRIPTION="A simple expenses tracker bot. Use /help to see commands."

CMD ["python", "-m", "app.main"]
