
import cv2
import numpy as np
import pandas as pd
import os
import time

# Глобальные переменные для хранения точек
points = []

# Функция для обработки кликов мыши
def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Точка {len(points)}: ({x}, {y})")
            cv2.circle(img_resized, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", img_resized)

# Функция для выравнивания изображения
def align_image(image, src_points, width, height):
    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    aligned = cv2.warpPerspective(image, M, (width, height))
    return aligned, M

# Основная функция
def process_image(image_path, output_folder):
    global points, img, img_resized

    # Загрузка изображения
    print(f"   - Загрузка изображения: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"   - Ошибка: Не удалось загрузить изображение {image_path}.")
        return None
    print(f"   - Изображение загружено: {image_path}")

    # Масштабирование изображения для отображения в фиксированном окне
    scale_percent = 50  # Масштаб в процентах
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    print(f"   - Изображение масштабировано: ({width}, {height})")


    # Отображение изображения и сбор точек
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", width, height)
    cv2.imshow("Image", img_resized)
    cv2.setMouseCallback("Image", click_event)
    print("Отметьте 4 угла листа бумаги (по часовой стрелке или против).")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 4:
        print("   - Ошибка: Необходимо отметить 4 точки.")
        return None
    print(f"   - Отмечено 4 точки: {points}")


    # Преобразование точек в формат numpy с учетом масштабирования
    src_points = np.float32(points) / (scale_percent / 100)
    print(f"   - Точки преобразованы: {src_points}")

    # Выравнивание изображения
    height, width = img.shape[:2]
    print(f"   - Выравнивание изображения")
    aligned_image, M = align_image(img, src_points, width, height)
    print(f"   - Выравнивание изображения завершено")

    # Сохранение выровненного изображения
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, aligned_image)
    print(f"   - Выровненное изображение сохранено как: {output_path}")

    # Возвращаем параметры гомографии
    homography_params = M.flatten()[:8]  # Первые 8 параметров матрицы
    print(f"   - Параметры гомографии: {homography_params}")

    return homography_params

# Папки для входных и выходных данных
input_folder = "peper/input_images"  # Папка с исходными изображениями
output_folder = "peper/aligned_images"  # Папка для выровненных изображений
csv_file = "peper/homography_data.csv"  # Файл для сохранения данных

# Создание папок, если они не существуют
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.dirname(csv_file), exist_ok=True)
print(f"CSV file path: {os.path.abspath(csv_file)}")


# Проверка и создание CSV-файла, если он не существует
if not os.path.exists(csv_file):
    print(f"CSV-файл {csv_file} не найден. Создаю новый файл.")
    df = pd.DataFrame(columns=["image_path", "h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32"])
    df.to_csv(csv_file, index=False)
    print(f"Создан новый CSV-файл: {csv_file}")
else:
        print(f"CSV-файл {csv_file} уже существует.")
# Обработка всех изображений в папке
print("Начало обработки изображений...")
csv_data = []
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    print(f"Обработка изображения: {image_name}")

    # Пропускаем не-изображения
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        print(f"Пропуск файла {image_name}: не является изображением.")
        continue
    print(f"  - Обрабатывается изображение: {image_name}")

    # Обработка изображения
    print("   - Начало обработки")
    homography_params = process_image(image_path, output_folder)
    print("   - Обработка завершена")

    if homography_params is not None:
        # Добавляем данные в CSV
        csv_data.append([image_path] + list(homography_params))
        print(f"  - Данные для {image_name} добавлены в csv_data.")
        print(f"  - csv_data after append: {csv_data}")

        # Сохранение данных в CSV
        print("  - Запись данных в CSV...")
        try:
             df = pd.DataFrame(csv_data, columns=["image_path", "h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32"])
             df.to_csv(csv_file, mode='w', header=True, index=False)
             print(f"  - Данные записаны в CSV: {csv_file}")
             csv_data.clear()
        except Exception as e:
            print(f"  - Ошибка при записи в CSV: {e}")
            
        while True:
            try:
                with open(csv_file, 'r') as test_file:
                        # Проверка успешной записи в файл
                    if not test_file.readline():
                        print('Ошибка записи в csv файл. Попытка повторной записи')
                        df = pd.DataFrame(csv_data, columns=["image_path", "h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32"])
                        df.to_csv(csv_file, mode='w', header=True, index=False)
                        print(f"  - Данные записаны в CSV: {csv_file}")
                        csv_data.clear()
                    else:
                        break
            except Exception as e:
                print(f"  - Ошибка проверки CSV файла: {e}")
                time.sleep(1)


    else:
        print(f"  - Ошибка при обработке изображения {image_name}. Данные не добавлены.")

    points.clear() # Очистка списка точек после каждого изображения
    print(f"  - Очистка списка точек")
print("Завершение обработки всех изображений")
