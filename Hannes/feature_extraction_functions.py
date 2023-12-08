import cv2 as cv
import numpy as np

def mean_colours(f_img, f_excel_layout, f_count):
    canny = cv.Canny(f_img, 50, 100)
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Finde Konturen im Bild
    largest_contour = max(contours, key=cv.contourArea)                             # Identifiziere die größte geschlossene Kontur
    x, y, w, h = cv.boundingRect(largest_contour)                                   # Extrahiere die Region of Interest (ROI) basierend auf der größten Kontur
    cv.rectangle(f_img, (x, y), (x + w, y + h), (0, 255, 0), 2)                       # Grüne Farbe (0, 255, 0), Dicke 2
    #cv.imshow(f'img{f_count}', f_img)

    roi = f_img[y:y+h, x:x+w]                     # Cropping
    average_color = np.mean(roi, axis=(0, 1))   # Berechne die durchschnittlichen RGB-Farbwerte der Pixel in der ROI

    average_blue =average_color[0]
    average_green =average_color[1]
    average_red =average_color[2]

    roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    average_hue =average_color[0]
    return average_blue, average_green, average_red, average_hue


    
