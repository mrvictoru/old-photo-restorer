FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8964

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8964"]