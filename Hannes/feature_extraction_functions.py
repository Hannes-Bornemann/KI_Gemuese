import cv2 as cv
import numpy as np
import os

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
    # Kopiere das Originalbild, um es nicht zu verändern
    image_copy = image.copy()
    img1 = cv.Canny(image_copy, 100, 50) #edges
    contours, hierarchy = cv.findContours(img1, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    contour_number = int(len(contours))
    # print("Anzahl der Konturen: ", contour_number)
    cv.drawContours(image_copy, contours, -1, (0, 255, 0), 1)  # Grüne Farbe, Linienbreite 2

    # # Bild vergrößert anzeigen lassen
    # image = cv.resize(image, zoom)
    # cv.imshow(f'img{count}', image)

    return contour_number

def mean_colours(image, count, zoom):

    image_copy = image.copy()
    img1 = cv.Canny(image_copy, 100, 50) #edges
    # finde Konturen, speichere sie in Liste und gebe Anzahl aus
    contours, hierarchy = cv.findContours(img1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # print("Anzahl der Konturen: ", len(contours))
    # finde die größte Kontur
    largest_contour = max(contours, key=cv.contourArea) 
    # finde konvexe Hülle
    hull = cv.convexHull(largest_contour)
    # Zeichne konvexe Hülle auf das Bild
    cv.drawContours(image_copy, [hull], 0, (0, 255, 0), 1)  # Grüne Farbe, Linienbreite 2

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
    image_copy = cv.resize(image_copy, zoom)
    cv.imshow(f'img{count}', image_copy)

    return average_blue, average_green, average_red, average_hue

def extent(image, count, zoom):

    image_copy = image.copy()
    img1 = cv.Canny(image_copy, 100, 50) #edges
    contours, hierarchy = cv.findContours(img1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea) 
    hull = cv.convexHull(largest_contour)

    rotated_rect = cv.minAreaRect(hull)

    # RotatedRect zeichnen
    box_points = cv.boxPoints(rotated_rect).astype(int)

    cv.drawContours(image_copy, [box_points], 0, (0, 255, 0), 1)

    # Bild vergrößert anzeigen lassen
    # image_copy = cv.resize(image_copy, zoom)
    # cv.imshow(f'img{count}', image_copy)

    w= np.linalg.norm(box_points[0] - box_points[1])
    h = np.linalg.norm(box_points[1] - box_points[2])
    longer_side = max(w, h)
    shorter_side = min(w,h)
    A_box = w*h
    A_mask = cv.contourArea(hull)

    extent = A_mask / A_box

    print("Fläche der Box in pixel: ", A_box, "Fläche des Objekts: ", A_mask, "Ausdehnung Extent: ", extent)

    aspect_ratio = shorter_side/longer_side
    print("Seitenverhältnis Aspect Ratio: ", aspect_ratio)

    return extent, aspect_ratio

def get_Features(image, count, zoom):
     # Kopiere das Originalbild, um es nicht zu verändern
    image_copy = image.copy()

    # contour_number
    img1 = cv.Canny(image_copy, 100, 50) # erzeugt einkanaliges Bild mit Kanten drauf
    contours, hierarchy = cv.findContours(img1, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # zählt Kanten, speichert sie in Liste
    contour_number = int(len(contours)) # Zählen der Listeneinträge
    # cv.drawContours(image_copy, contours, -1, (0, 255, 0), 1)  # Alle Konturen auf bild zeichnen

    # aspect ratio, extent
    largest_contour = max(contours, key=cv.contourArea) 
    hull = cv.convexHull(largest_contour)
    cv.drawContours(image_copy, [hull], 0, (0, 255, 0), 1)  # konvexe Hülle auf Bild zeichnen
    rotated_rect = cv.minAreaRect(hull)
    box_points = cv.boxPoints(rotated_rect).astype(int)
    cv.drawContours(image_copy, [box_points], 0, (0, 255, 0), 1)    # Bounding Box auf Bild zeichnen
    w= np.linalg.norm(box_points[0] - box_points[1])
    h = np.linalg.norm(box_points[1] - box_points[2])
    longer_side = max(w, h)
    shorter_side = min(w,h)
    A_box = w*h
    A_mask = cv.contourArea(hull)
    extent = A_mask / A_box
    aspect_ratio = shorter_side/longer_side

    # mean colours
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)   # leere Maske erzeugen
    cv.fillPoly(mask, [hull], (255))    # ausgefüllte hülle in maske schreiben
    pixel_in_mask = image_copy[mask == 255]    # Maske über Bild legen und relevante Pixel auswählen
    average_blue = np.mean(pixel_in_mask[:, 0])   #  Farbmittelwerte für Kanäle BGR in der Maske berechnen
    average_green = np.mean(pixel_in_mask[:, 1])
    average_red = np.mean(pixel_in_mask[:, 2])
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)   # Farbmittelwert für Hue Kanal in der Maske berechnen
    pixel_in_mask_hsv = hsv[mask == 255]
    average_hue = np.mean(pixel_in_mask_hsv[:, 0])

    # Bild vergrößert anzeigen lassen
    image_copy = cv.resize(image_copy, zoom)
    cv.imshow(f'img{count}', image_copy)

    return contour_number, aspect_ratio, extent, average_blue, average_green, average_red , average_hue
    
