FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir fastapi uvicorn pydantic

CMD ["python", "server.py"]