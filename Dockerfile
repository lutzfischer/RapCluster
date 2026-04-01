# Stage 1: Build React frontend
FROM node:20 AS frontend-build

WORKDIR /app
COPY clustering-frontend/package*.json ./
RUN npm ci
COPY clustering-frontend/ ./
RUN npm run build

# Stage 2: Combined runtime
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    nginx \
    supervisor \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY supervisord.conf /etc/supervisor/conf.d/app.conf

# Remove default nginx site to avoid conflicts
RUN rm -f /etc/nginx/sites-enabled/default

RUN useradd --no-create-home --shell /bin/false appuser \
 && chown -R appuser:appuser /app \
 && chown -R www-data:www-data /usr/share/nginx/html

EXPOSE 8080

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/app.conf"]
