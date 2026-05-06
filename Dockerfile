FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml requirements.txt ./

RUN uv pip install --system -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7860"]
