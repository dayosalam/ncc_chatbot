FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
COPY ./packages/pirate-speak /app/packages/pirate-speak

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir -e /app/packages/pirate-speak

COPY ./app /app/app

EXPOSE 8080

CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8080
