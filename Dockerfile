# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set workdir
WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y build-essential git curl unzip && \
    apt-get clean

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject & install deps
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Copy rest of the app
COPY . .

# Expose necessary ports
EXPOSE 5000 8501 8000 9000 8080

# Default command
CMD ["poetry", "run", "python", "-m", "src.main"]
