FROM python:3.11-slim
WORKDIR /app
COPY analysis/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared /app/services/shared
COPY analysis/app /app/app
ENV PYTHONPATH=/app/services/shared:$PYTHONPATH
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
