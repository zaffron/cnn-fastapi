FROM tensorflow/tensorflow:2.17.0-gpu

WORKDIR /app

RUN apt-get update && apt-get install -y \
  git \
  wget \
  unzip \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p app/data/log_files && touch app/data/log_files/logs.logs
RUN mkdir -p app/model-artifacts

EXPOSE 8000

CMD ["python", "main.py"]
