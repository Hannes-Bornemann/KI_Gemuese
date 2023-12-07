import cv2 as cv
import numpy as np

img = cv.imread('photos/Zwiebel/Zwiebel (13).jpg')
img = cv.resize(img, (800, 600))

canny = cv.Canny(img, 125, 175)         # 125, 175 war gut, sind schwellenwerte, drunter= keine kante, drüber = kante, dazwischen = kante wenn nachbarpixel kante
cv.imshow('canny', canny)

# Finde Konturen im Bild
contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Identifiziere die größte geschlossene Kontur
largest_contour = max(contours, key=cv.contourArea)

# Extrahiere die Region of Interest (ROI) basierend auf der größten Kontur
x, y, w, h = cv.boundingRect(largest_contour)
cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Grüne Farbe (0, 255, 0), Dicke 2
cv.imshow('img', img)
roi = img[y:y+h, x:x+w]

# Berechne die durchschnittlichen RGB-Farbwerte der Pixel in der ROI
average_color = np.mean(roi, axis=(0, 1))

# Ausgabe der durchschnittlichen Farbwerte
print("Durchschnittliche Farbwerte (BGR):", average_color)

cv.waitKey(0)
cv.destroyAllWindows() 