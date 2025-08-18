FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8964

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8964"]