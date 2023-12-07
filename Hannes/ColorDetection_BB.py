import cv2 as cv
import numpy as np
import pandas as pd

excel_layout = {
        'Blue': [],
        'Green': [],
        'Red': [],
        'Hue': [],
    }

for i in range (1,2):

    path = f'photos/Zwiebel/Zwiebel ({i}).jpg'

    img = cv.imread(path)
    img = cv.resize(img, (800, 600))

    canny = cv.Canny(img, 50, 100)         # 125, 175 war gut, sind schwellenwerte, drunter= keine kante, drüber = kante, dazwischen = kante wenn nachbarpixel kante
    #cv.imshow('canny', canny)


    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Finde Konturen im Bild
    largest_contour = max(contours, key=cv.contourArea)                             # Identifiziere die größte geschlossene Kontur
    x, y, w, h = cv.boundingRect(largest_contour)                                   # Extrahiere die Region of Interest (ROI) basierend auf der größten Kontur
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)                       # Grüne Farbe (0, 255, 0), Dicke 2
    #cv.imshow(f'img{i}', img)

    roi = img[y:y+h, x:x+w]                     # Cropping
    average_color = np.mean(roi, axis=(0, 1))   # Berechne die durchschnittlichen RGB-Farbwerte der Pixel in der ROI

    average_blue =average_color[0]
    average_green =average_color[1]
    average_red =average_color[2]

    roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    average_hue =average_color[0]

    # print("Durchschnittliche Farbwerte (BGR):")  # Ausgabe der durchschnittlichen Farbwerte
    value_bLue = excel_layout['Blue']
    value_bLue.append(average_blue)
    value_green = excel_layout['Green']
    value_green.append(average_green)
    value_red = excel_layout['Red']
    value_red.append(average_red)
    value_hue = excel_layout['Hue']
    value_hue.append(average_hue)

    #print(average_blue)
    #print(average_green)
    #print(average_red)

    df = pd.DataFrame(excel_layout)   #.T transponiert Dataframe, sodass RGB Werte in Excel nicht untereinander, sonder nebeneinander geschrieben werden
    df.to_excel('output.xlsx', index=False, startrow=0, startcol=0)

cv.waitKey(0)
cv.destroyAllWindows() 