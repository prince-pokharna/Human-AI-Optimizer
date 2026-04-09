FROM python:3.10

WORKDIR /app

COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Start the Gradio app (UI)
CMD ["python", "server/app.py"]