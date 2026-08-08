FROM python:3.10-slim

WORKDIR /app

# install dependencies first so docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all application files and the docs folder
COPY . .

# expose the port the app will listen on
EXPOSE 7860

# start the fastapi app with uvicorn
# zepto_app is the FastAPI instance defined in app.py
CMD ["uvicorn", "app:zepto_app", "--host", "0.0.0.0", "--port", "7860"]
