FROM tiangolo/uvicorn-gunicorn:python3.9

USER root

COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY data /app/data

COPY api/app/ /app