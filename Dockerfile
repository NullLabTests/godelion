FROM python:3.10-slim

LABEL org.opencontainers.image.title="Godelion"
LABEL org.opencontainers.image.description="Open-Ended Evolution of Self-Improving Coding Agents"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /godelion

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]
