FROM python:3.10-slim

WORKDIR /app

# copy requirements first so docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the application files and the docs folder
COPY . .

# expose the port the app will run on
EXPOSE 7860

# start the fastapi app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
