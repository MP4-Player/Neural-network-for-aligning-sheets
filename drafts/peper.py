import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('peper/31.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Поиск контуров
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

# Аппроксимация контура
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# Проверка, что найден прямоугольник
if len(approx) == 4:
    # Упорядочивание углов
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    approx = order_points(approx.reshape(4, 2))

    # Вычисление матрицы преобразования
    (tl, tr, br, bl) = approx
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(approx, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    # Сохранение результата
    cv2.imwrite('aligned_paper_with_drawing.jpg', warped)
    print("Лист с рисунком выровнен и сохранен как 'aligned_paper_with_drawing.jpg'")
else:
    print("Не удалось найти четыре угла листа.")