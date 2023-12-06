FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn
EXPOSE 5000
ENTRYPOINT ["gunicorn", "flk.py", "-b", "0.0.0.0:5000"]

