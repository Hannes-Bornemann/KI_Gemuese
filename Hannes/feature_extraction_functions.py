import cv2 as cv
import numpy as np
import os

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

def resize(input_dir, output_dir, file, new_width, new_height):
    # Bilder Verkleinern und in Ordner speichern
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, file)
    img = cv.imread(input_path) # Bild ist 3:4
    height, width = img.shape[:2]

    # print ("höhe: ", height, "breite: ", width, "höhe/breite: ", width/height)
    
    # Wenn bild in Breite erweitert werden muss:
    if width/height < new_width/new_height:
        enlargement = int((height * (new_width/new_height) - width) / 2)
        up = 0
        down = 0
        left = enlargement
        right = enlargement
    # Wenn bild in Höhe erweitert werden muss:
    elif width/height > new_width/new_height:
        enlargement = int((width / (new_width/new_height) - height) / 2)
        up = enlargement
        down = enlargement
        left = 0
        right = 0
    # Wenn Bild nicht erweitert werden muss:
    else:
        up, down, left, right = (0,0,0,0)
    img_enlarged = cv.copyMakeBorder(img, up, down, left, right, cv.BORDER_REPLICATE) # Altenativ: ...,cv.BORDER_CONSTANT, value=[0, 0, 0]

    img = cv.resize(img_enlarged, (new_width, new_height))

    # Verkleinertes Bild in Ordner speichern
    cv.imwrite(output_path, img)
    # Verkleinertes Bild zurück geben
    return img

def contour(image):
    # img = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #gray
    #img1 = cv.convertScaleAbs(image, alpha=1.3, beta=20) #Versuch, Schatten zu verringern
    #img1 = cv.GaussianBlur(image, (3,3), cv.BORDER_DEFAULT)
    img1 = cv.Canny(image, 100, 50) #edges
    
    # img = cv.dilate(img, None, iterations=6)
    # finde Konturen, speichere sie in Liste und gebe Anzahl aus
    contours, hierarchy = cv.findContours(img1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print("Anzahl der Konturen: ", len(contours))
    # finde die größte Kontur
    largest_contour = max(contours, key=cv.contourArea) 
    hull = cv.convexHull(largest_contour)
    # Zeichne die größte Kontur auf das Bild
    cv.drawContours(image, [hull], 0, (0, 255, 0), 1)  # Grüne Farbe, Linienbreite 2
    
    return image

def contour_number(image):
    img1 = cv.Canny(image, 100, 50) #edges
    contours, hierarchy = cv.findContours(img1, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    print("Anzahl der Konturen: ", len(contours))
    cv.drawContours(image, contours, -1, (0, 255, 0), 1)  # Grüne Farbe, Linienbreite 2

    return image

