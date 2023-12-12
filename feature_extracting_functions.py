import cv2
import numpy as np
import matplotlib.pyplot as plt

def contour(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
        
        #Perform edge detection
        #edges = cv2.Canny(imgBlur, 120, 150)
        edges = cv2.Canny(imgBlur, 1, 20)
        
        # cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
        # cv2.imshow("Canny", edges)
        # cv2.imwrite("Canny.png", edges)
        
        #edges = cv2.Canny(imgBlur, 0, 60) 
        #edges = cv2.Canny(imgBlur, 150, 300) 
        edges = cv2.dilate(edges, None, iterations=6)
        
        cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
        cv2.imshow("Canny", edges)
        cv2.resizeWindow("Canny", 800, 600)
        #cv2.imwrite("Canny.png", edges)
        
        #Find the contours in the image
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        print("Anzahl der Konturen: ", len(contours))
        
        contours = list(contours)
        if contours:
            # contour with the highest x-value
            #largest_contour = max(contours, key=lambda contour: cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2])
            contours.sort(key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]
        
            # Extract the bounding box
            x_min, y_min, w, h = cv2.boundingRect(largest_contour)

            # Kopiere das Originalbild, um es nicht zu verändern
            result_image = image.copy()
            
            # Zeichne die größte Kontur auf das Bild
            cv2.drawContours(result_image, [largest_contour], 0, (0, 255, 0), 2)  # Grüne Farbe, Linienbreite 2

            # Zeichne alle Konturen auf das Bild
            #for contour in contours:
             #   cv2.drawContours(result_image, [contour], 0, (255, 0, 0), 2)  # Blaue Farbe, Linienbreite 2

            # Zeichne die Bounding Box auf das Bild
            cv2.rectangle(result_image, (x_min, y_min), (x_min + w, y_min + h), (0, 0, 255), 2)  # Rote Farbe, Linienbreite 2
            
            # Zeige das Bild mit der größten Kontur an
            cv2.namedWindow("Largest Contour", cv2.WINDOW_NORMAL)
            cv2.imshow("Largest Contour", result_image)
            cv2.resizeWindow("Largest Contour", 800, 600)
        
        else:
            print("Keine Konturen gefunden.")
            
        # create an empty mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # fill the contour into the mask
        cv2.fillPoly(mask, [largest_contour], (255))

        cv2.waitKey(0)

        # Fläche des Objekts (Kontur)
        contour_area = cv2.contourArea(max(contours, key=cv2.contourArea))

        
        return mask, largest_contour

def extent(image, contours):
    rotated_rect = cv2.minAreaRect(contours)

    # RotatedRect zeichnen
    box_points = cv2.boxPoints(rotated_rect).astype(int)

    cv2.drawContours(image, [box_points], 0, (0, 255, 0), 2)
    cv2.namedWindow("Rotated Rect um das Objekt", cv2.WINDOW_NORMAL)
    cv2.imshow("Rotated Rect um das Objekt", image)
    cv2.resizeWindow("Rotated Rect um das Objekt", 800, 600)
    cv2.waitKey(0)

    w= np.linalg.norm(box_points[0] - box_points[1])
    h = np.linalg.norm(box_points[1] - box_points[2])

    A_box = w*h
    A_mask = cv2.contourArea(contours)

    extent = A_mask / A_box

    print("Fläche der Box in pixel: ", A_box, "Fläche des Objekts: ", A_mask, "Ausdehnung Extent: ", extent)

    aspect_ratio = w/h
    print("Seitenverhältnis Aspect Ratio: ", aspect_ratio)

    return box_points, extent, aspect_ratio

def colour_hist(image, mask):
    # Farbhistogramm für das Bild mit der Maske erstellen
    histogramm = cv2.calcHist([image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])


    # Histogramm normalisieren, falls gewünscht
    #histogramm = cv2.normalize(histogramm, histogramm)

    # Daten des Histogramms in ein eindimensionales Array umformen
    histogramm = histogramm.flatten()

    #print(histogramm)

    # Histogramm anzeigen (optional)
    plt.plot(histogramm)
    plt.title('Farbhistogramm für das Bild mit Maske')
    plt.show()
    
    return histogramm

def mean_colours(image, mask):
     # Pixel in der Maske auswählen
    pixels_in_maske = image[mask == 255]

    # Farbmittelwert für jedes Kanal in der Maske berechnen
    durchschnitt_rot = np.mean(pixels_in_maske[:, 0])
    durchschnitt_gruen = np.mean(pixels_in_maske[:, 1])
    durchschnitt_blau = np.mean(pixels_in_maske[:, 2])

    # Farbmittelwerte zurückgeben
    return durchschnitt_rot, durchschnitt_gruen, durchschnitt_blau
