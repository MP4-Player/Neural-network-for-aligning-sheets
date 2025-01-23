import cv2
import os

# Путь к видеофайлу
video_path = 'video_2025-01-20_20-09-37.mp4'

# Папка для сохранения кадров
output_folder = 'output_folder'
os.makedirs(output_folder, exist_ok=True)

# Открываем видео
cap = cv2.VideoCapture(video_path)

# Проверяем, удалось ли открыть видео
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео.")
    exit()

frame_count = 0
saved_frame_count = 0

while True:
    # Читаем кадр из видео
    ret, frame = cap.read()

    # Если кадр не удалось прочитать, выходим из цикла
    if not ret:
        break

    # Сохраняем каждый второй кадр
    if frame_count % 2 == 0:
        frame_filename = os.path.join(output_folder, f'8frame_{saved_frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    frame_count += 1

# Освобождаем ресурсы
cap.release()

print(f"Сохранено {saved_frame_count} кадров в папку '{output_folder}'.")