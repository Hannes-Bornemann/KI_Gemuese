import cv2 as cv
import numpy as np

def mean_colours(f_img, f_excel_layout, f_count):
    canny = cv.Canny(f_img, 50, 100)
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Finde Konturen im Bild
    largest_contour = max(contours, key=cv.contourArea)                             # Identifiziere die größte geschlossene Kontur
    x, y, w, h = cv.boundingRect(largest_contour)                                   # Extrahiere die Region of Interest (ROI) basierend auf der größten Kontur
    cv.rectangle(f_img, (x, y), (x + w, y + h), (0, 255, 0), 2)                     # Grüne Farbe (0, 255, 0), Dicke 2
    #cv.imshow(f'img{f_count}', f_img)

    roi_bgr = f_img[y:y+h, x:x+w]                     # Cropping
    roi_hsv = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV)
    average_bgr = np.mean(roi_bgr, axis=(0, 1))   # Berechne die durchschnittlichen BGR-Farbwerte der Pixel in der ROI
    average_hsv = np.mean(roi_hsv, axis=(0, 1))

    average_blue =average_bgr[0]
    average_green =average_bgr[1]
    average_red =average_bgr[2]
    average_hue =average_hsv[0]

    return f_img, average_blue, average_green, average_red, average_hue



    
