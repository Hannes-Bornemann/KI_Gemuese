import cv2 as cv
import numpy as np
import os
"""
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
"""
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

def contour_number(image, count, zoom):
    img1 = cv.Canny(image, 100, 50) #edges
    contours, hierarchy = cv.findContours(img1, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    
    contour_number = int(len(contours))
    print("Anzahl der Konturen: ", contour_number)
    cv.drawContours(image, contours, -1, (0, 255, 0), 1)  # Grüne Farbe, Linienbreite 2

    # Bild vergrößert anzeigen lassen
    # image = cv.resize(image, zoom)
    # cv.imshow(f'img{count}', image)

    return contour_number

def mean_colours(image, count, zoom):

    img1 = cv.Canny(image, 100, 50) #edges
    # finde Konturen, speichere sie in Liste und gebe Anzahl aus
    contours, hierarchy = cv.findContours(img1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # print("Anzahl der Konturen: ", len(contours))
    # finde die größte Kontur
    largest_contour = max(contours, key=cv.contourArea) 
    # finde konvexe Hülle
    hull = cv.convexHull(largest_contour)
    # Zeichne konvexe Hülle auf das Bild
    cv.drawContours(image, [hull], 0, (0, 255, 0), 1)  # Grüne Farbe, Linienbreite 2

    # create an empty mask
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # fill the contour into the mask
    cv.fillPoly(mask, [hull], (255))
    
     # Pixel in der Maske auswählen
    pixels_in_maske = image[mask == 255]

    # Farbmittelwerte für Kanäle BGR in der Maske berechnen
    average_blue = np.mean(pixels_in_maske[:, 0])
    average_green = np.mean(pixels_in_maske[:, 1])
    average_red = np.mean(pixels_in_maske[:, 2])

    # Farbmittelwert für Hue Kanal in der Maske berechnen
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    pixels_in_maske_hsv = hsv[mask == 255]
    average_hue = np.mean(pixels_in_maske_hsv[:, 0])

    # Bild vergrößert anzeigen lassen
    mask = cv.resize(mask, zoom)
    cv.imshow(f'img{count}', mask)

    return average_blue, average_green, average_red, average_hue
