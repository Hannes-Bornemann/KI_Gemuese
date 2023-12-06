import cv2 as cv
import numpy as np

img = cv.imread('photos/Zwiebel/Zwiebel (2).jpg')
img = cv.resize(img, (800, 600))

# Konvertiere das Bild in den Graustufenmodus
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('gray_img', gray_img)

# Wende Canny-Kanten-Erkennung an
edges = cv.Canny(gray_img, 60, 120)

cv.imshow('edges', edges)

contourslist, hierarchies = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)    

                                                                                    # Gibt "contours" als list zurück, welche alle Koordinaten der Konturpunkte enthält
                                                                                    # Gibt "hierarchies" zurück, beschreibung ob bestimmte Konturen innerhalb von anderen liegen, z.b. kreis innerhalb rechteck
                                                                                    # "canny" = übergebenes Bild, 
                                                                                    # "cv.RETR_List"            =modus alle Konturen (TREE=alle hierarchischen Konturen, EXTERNAL=alle Außenkonturen)
                                                                                    # "cv.CHAIN_APPROX_SIMPLE"  =Approximationsmethode, None= keine, d.h. alle Pixel Koordinaten werden ausgegeben,
                                                                                    #  Simple= Reduzierung einer Linie auf Koord. von Anfangs- und Endpunkt
print(f'{len(contourslist)} contour(s) found!')

cv.drawContours(edges, contourslist, -1, (0,0,255), 1)  #(übergebenesBild, ÜbergebeneListe, KonturIndex= wvl. Konturen wollen wir auf dem Bild, -1 = alle, FarbeDerKonturen, DickeDerLinie )
cv.imshow('Contours', edges)


# # Finde Konturen im Bild
# contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# # Identifiziere die größte geschlossene Kontur
# largest_contour = max(contours, key=cv.contourArea)

# # Erzeuge eine leere Maske
# mask = np.zeros_like(gray_img)

# # Zeichne die äußere Kante auf die Maske (ohne Füllung)
# cv.drawContours(mask, [largest_contour], 0, 255, thickness=2)

# # Zeige das Bild mit der eingefärbten äußeren Kante
# result = cv.bitwise_and(img, img, mask=mask)
# cv.imshow('Image with Outer Contour', result)

cv.waitKey(0)
cv.destroyAllWindows()
