import cv2
import numpy as np

def auto_align_paper(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    orig = image.copy()
    
    # Преобразование в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применение GaussianBlur для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Адаптивная бинаризация для выделения границ
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Детекция границ с помощью Canny
    edged = cv2.Canny(binary, 75, 200)
    
    # Поиск контуров
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Поиск контура листа бумаги
    for contour in contours:
        # Аппроксимация контура
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Если контур имеет 4 вершины, предполагаем, что это лист бумаги
        if len(approx) == 4:
            screenCnt = approx
            break
    
    # Если не нашли контур с 4 вершинами, возвращаем исходное изображение
    if 'screenCnt' not in locals():
        print("Не удалось найти лист бумаги на изображении.")
        return orig
    
    # Отображение контура на изображении
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    
    # Применение гомографии для выравнивания листа
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def four_point_transform(image, pts):
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    # Применение преобразования
    warped = four_point_transform(orig, screenCnt.reshape(4, 2))
    
    return warped

# Пример использования
aligned_image = auto_align_paper("peper/extracted_frames/4frame_0103.jpg")
cv2.imshow("Aligned Image", aligned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





###############################'peper/extracted_frames/4frame_0103.jpg'0000__peper/input_images/1frame_0005.jpg