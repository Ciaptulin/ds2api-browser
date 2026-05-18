FROM python:3.11-slim

# Install system deps for Chromium
RUN apt-get update && apt-get install -y \
    curl wget ca-certificates \
    libnss3 libnspr4 libatk-bridge2.0-0 libatk1.0-0 \
    libcups2 libdrm2 libdbus-1-3 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libasound2 libpango-1.0-0 libcairo2 \
    fonts-liberation libu2f-udev libvulkan1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser (Chromium for headless)
RUN python -m playwright install chromium

# Copy app code
COPY . .

# HF Spaces uses port 7860
ENV PORT=7860
ENV DS2API_PORT=7860
ENV DS2API_HOST=127.0.0.1
ENV DS2API_HEADLESS=true
ENV DS2API_HUMANIZE=true
ENV DS2API_MAX_CONCURRENT=50
ENV DS2API_MAX_ACTIVE_BROWSERS=50
ENV DISPLAY=:99

# Start Xvfb + app
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 7860

CMD ["/start.sh"]