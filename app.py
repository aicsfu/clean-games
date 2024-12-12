import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import gradio as gr

# Задаем шрифт
font_path = "arial.ttf"  # Убедитесь, что файл шрифта доступен или замените на другой путь
font_size = 20
font = ImageFont.truetype(font_path, font_size)

# Загрузка модели и процессора
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to("cuda" if torch.cuda.is_available() else "cpu")

# Определяем текстовые запросы
text = "a trash. a bottle. a plastic bag. a can. a paper. juice box. a cardboard box. a food wrapper. a plastic container. a cup. a lid. a plastic cup. a food waste. a soda can. a napkin. a disposable utensil. a carton. a plastic wrapper. an aluminum can. a glass bottle. a plastic bottle. a tissue. a plastic. a wrapper."

def detect_objects_count_and_draw(image):
    # Конвертация изображения в RGB
    image = image.convert("RGB")

    # Обработка входных данных и выполнение обнаружения объектов
    inputs = processor(images=image, text=text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model(**inputs)

    # Пост-обработка результатов для получения ограничивающих рамок
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.27,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )

    # Нарисовать ограничивающие рамки и подписи
    draw = ImageDraw.Draw(image)
    detected_objects = set()
    
    for box, label in zip(results[0]["boxes"], results[0]["labels"]):
        box = box.cpu().tolist()
        draw.rectangle(box, outline="red", width=3)
        text_position = (box[0], box[1] - 10)
        draw.text(text_position, label, fill="black", font=font)
        detected_objects.add(label)
    
    return image, len(results[0]["boxes"])

def check_cleanup(image_before, image_after):
    # Получаем изображения с аннотациями и количество объектов до и после уборки
    image_before_annotated, count_before = detect_objects_count_and_draw(image_before)
    image_after_annotated, count_after = detect_objects_count_and_draw(image_after)

    # Проверка процента удаления объектов
    if count_before > 0 and ((count_before - count_after) / count_before) * 100 > 50:
        result_text = f"Уборка была проведена. Найдено объектов на первом изображении: {count_before}, на втором изображении: {count_after}."
    else:
        result_text = f"Уборка не была завершена или процент очищения менее 50%. Найдено объектов на первом изображении: {count_before}, на втором изображении: {count_after}."
    
    return image_before_annotated, image_after_annotated, result_text

# Создаем интерфейс Gradio
interface = gr.Interface(
    fn=check_cleanup,
    inputs=[gr.Image(type="pil", label="Изображение до уборки"), gr.Image(type="pil", label="Изображение после уборки")],
    outputs=[gr.Image(type="pil", label="Аннотированное изображение до уборки"), 
             gr.Image(type="pil", label="Аннотированное изображение после уборки"),
             gr.Textbox(label="Результат")],
    title="Проверка уборки мусора",
    description="Загрузите два изображения: одно до уборки, другое после. Программа определит, была ли уборка успешно завершена и выведет количество объектов на каждом изображении."
)

# Запуск интерфейса
print("Gradio запущен на порту 7860")
interface.launch(server_port=7860, server_name="0.0.0.0", share=False)
