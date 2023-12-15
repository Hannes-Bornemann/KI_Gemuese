import cv2 as cv
import pandas as pd
import numpy as np
import os
import feature_extraction_functions

excel_layout = {
    'contour number': [],
    'aspect ratio': [],
    'extent': [],
    'Blue': [],
    'Green': [],
    'Red': [],
    'Hue': [],
    }
input_dir = 'photos/Karotte'
output_dir = 'photos_reduced/Karotte_reduced'
files = os.listdir(input_dir)   # speichert dateinamen in liste

new_width, new_height = (125, 125) # (gewünschte Auflösung (Breite, Höhe) des bildes für die Merkmalerkennung)
zoomfactor = int(600 / max(new_width, new_height)) # zahl 600 ist die größte seite des Anzeigefensters
zoom = (new_width*zoomfactor, new_height*zoomfactor) # Berechnung des Zoomfaktors (nur für Anzeige)

max_files = 20  # Anzahl bilder die gelesen werden sollen
count = 1

for file in files:
    # Bild aus Ordner einlesen, verkleinern in Zielordner schreiben und zurück geben
    img = feature_extraction_functions.resize(input_dir, output_dir, file, new_width, new_height)

    # finale Merkmale:
    # contour_number = feature_extraction_functions.contour_number(img, count, zoom)
    # average_blue, average_green, average_red, average_hue = feature_extraction_functions.mean_colours(img, count, zoom)
    # extent, aspect_ratio = feature_extraction_functions.extent(img, count, zoom)

    contour_number, aspect_ratio, extent, average_blue, average_green, average_red , average_hue = feature_extraction_functions.get_Features(img, count, zoom)

    # werte an value im dictionary hängen
    excel_layout['contour number'].append(contour_number)
    excel_layout['aspect ratio'].append(aspect_ratio)
    excel_layout['extent'].append(extent)
    excel_layout['Blue'].append(average_blue)
    excel_layout['Green'].append(average_green)
    excel_layout['Red'].append(average_red)
    excel_layout['Hue'].append(average_hue)
    
    if count >= max_files:
        break
    count += 1
    

df = pd.DataFrame(excel_layout) 
df.to_csv('output.csv', sep=';') # csv speichert weniger Metadaten, braucht weniger Speicher, sehr sinnvoll wenn Daten iwann sehr groß werden 

cv.waitKey(0)
cv.destroyAllWindows() 
