import cv2 as cv
import numpy as np
import pandas as pd

img = cv.imread('photos/Zwiebel/Zwiebel (13).jpg')
img = cv.resize(img, (800, 600))

canny = cv.Canny(img, 125, 175)         # 125, 175 war gut, sind schwellenwerte, drunter= keine kante, drüber = kante, dazwischen = kante wenn nachbarpixel kante
cv.imshow('canny', canny)
contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Finde Konturen im Bild
largest_contour = max(contours, key=cv.contourArea)                             # Identifiziere die größte geschlossene Kontur
x, y, w, h = cv.boundingRect(largest_contour)                                   # Extrahiere die Region of Interest (ROI) basierend auf der größten Kontur
cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)                       # Grüne Farbe (0, 255, 0), Dicke 2
cv.imshow('img', img)
roi = img[y:y+h, x:x+w]
average_color = np.mean(roi, axis=(0, 1))                   # Berechne die durchschnittlichen RGB-Farbwerte der Pixel in der ROI
print("Durchschnittliche Farbwerte (BGR):", average_color)  # Ausgabe der durchschnittlichen Farbwerte

df = pd.DataFrame(average_color).T      #.T transponiert Dataframe, sodass RGB Werte in Excel nicht untereinander, sonder nebeneinander geschrieben werden
df.to_excel('output.xlsx', index=False, startrow=2, startcol=2)

cv.waitKey(0)
cv.destroyAllWindows() 