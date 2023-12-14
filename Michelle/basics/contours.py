import cv2 as cv
import numpy as np


img = cv.imread('photos/Zwiebel/Zwiebel (2).jpg')
img = cv.resize(img, (800, 600))

#cv.imshow('Zwiebel (2)', img)

blank= np.zeros(img.shape, dtype='uint8')
cv.imshow('blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('gray', gray)
#
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

canny = cv.Canny(img, 125, 175)
cv.imshow('canny', canny)

# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY) # Macht Bild binär, d.h. wenn Wert über threshold 125 (Grenzwert), dann wird Pixel weiß (1) sonst zu schwarz (0)
# cv.imshow('thresh', thresh)

contourslist, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)    

                                                                                    # Gibt "contours" als list zurück, welche alle Koordinaten der Konturpunkte enthält
                                                                                    # Gibt "hierarchies" zurück, beschreibung ob bestimmte Konturen innerhalb von anderen liegen, z.b. kreis innerhalb rechteck
                                                                                    # "canny" = übergebenes Bild, 
                                                                                    # "cv.RETR_List"            =modus alle Konturen (TREE=alle hierarchischen Konturen, EXTERNAL=alle Außenkonturen)
                                                                                    # "cv.CHAIN_APPROX_SIMPLE"  =Approximationsmethode, None= keine, d.h. alle Pixel Koordinaten werden ausgegeben,
                                                                                    #  Simple= Reduzierung einer Linie auf Koord. von Anfangs- und Endpunkt
print(f'{len(contourslist)} contour(s) found!')

cv.drawContours(blank, contourslist, -1, (0,0,255), 1)  #(übergebenesBild, ÜbergebeneListe, KonturIndex= wvl. Konturen wollen wir auf dem Bild, -1 = alle, FarbeDerKonturen, DickeDerLinie )
cv.imshow('Contours', blank)

cv.waitKey(0)
cv.destroyAllWindows() 

# Learning:
# - zuerst Canny nutzen
# - wenn das nicht klappt dann cv.findContoures (mit Hierarchies)
# - wenn das nicht klappt, dann erst threshold, also binärisieren
