# Базовый образ с поддержкой Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем зависимости для PyTorch (CUDA поддержка)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости и код
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY arial.ttf arial.ttf

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Загружаем предварительно модели для экономии времени
RUN python -c "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; \
    AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); \
    AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny')"

# Указываем порт
EXPOSE 7860

RUN ls -l /app

# Запускаем приложение
CMD ["python", "app.py"]
