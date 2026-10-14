# Image de l'API de scoring.
# Prérequis : avoir entraîné le modèle (models/lgbm.joblib) avant de construire l'image.
#   docker build -t credit-scoring-api .
#   docker run -p 8000:8000 credit-scoring-api
FROM python:3.12-slim

# LightGBM a besoin de la bibliothèque OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
WORKDIR /app

# Dépendances d'abord : cette couche reste en cache tant que pyproject.toml et uv.lock ne changent pas
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY models/lgbm.joblib ./models/lgbm.joblib
RUN uv sync --frozen --no-dev

ENV MODEL_PATH=/app/models/lgbm.joblib
EXPOSE 8000
# Utilisateur sans droits d'administration
RUN useradd --create-home api && chown -R api /app
USER api

HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uv", "run", "--no-sync", "uvicorn", "credit_risk_scoring.api:app", "--host", "0.0.0.0", "--port", "8000"]
