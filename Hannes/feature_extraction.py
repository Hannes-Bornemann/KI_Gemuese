import cv2 as cv
import pandas as pd
import numpy as np
import os
import feature_extraction_functions

df_layout = {
    "contour number": [],
    "aspect ratio": [],
    "extent": [],
    "Blue": [],
    "Green": [],
    "Red": [],
    "Hue": [],
    "class": [],
}
df = pd.DataFrame(df_layout)  # Erstellt Dataframe layout
# gewünschte Auflösung (Breite, Höhe) des bildes für die Merkmalerkennung
new_width, new_height = (125, 125)
# zahl 600 ist die größte seite des Anzeigefensters
zoomfactor = int(600 / max(new_width, new_height))
# Berechnung des Zoomfaktors (nur für Anzeige)
zoom = (new_width * zoomfactor, new_height * zoomfactor)


for i in range(3):  # for Schleife auf Bilder aus verschiedenen Ordnern zuzugreifen
    if i == 0:
        input_dir = "photos/Zwiebel"
        output_dir = "photos_reduced/Zwiebel_reduced"
        files = os.listdir(input_dir)  # speichert dateinamen in liste
        count = 1  # setzten
        max_files = 50  # Anzahl bilder die gelesen werden sollen
    elif i == 1:
        input_dir = "photos/Karotte"
        output_dir = "photos_reduced/Karotte_reduced"
        files = os.listdir(input_dir)
        count = 51  # zurücksetzten (nur Reihe fortsetzen wenn bilder angezeigt werden sollen, weil bilder sich sonst überschreiben. Sonst kann wieder bei 1 begoonen werden und immer gleiches "max_files" genutzt werden)
        max_files = 100  # Anzahl bilder die gelesen werden sollen
    elif i == 2:
        input_dir = "photos/Kartoffel"
        output_dir = "photos_reduced/Kartoffel_reduced"
        files = os.listdir(input_dir)
        count = 101
        max_files = 150

    for file in files:
        # Bild aus Ordner einlesen, verkleinern in Zielordner schreiben und zurück geben
        img = feature_extraction_functions.resize(
            input_dir, output_dir, file, new_width, new_height
        )

        (
            contour_number,
            aspect_ratio,
            extent,
            average_blue,
            average_green,
            average_red,
            average_hue,
        ) = feature_extraction_functions.get_Features(img, count, zoom)

        # werte an value im dictionary hängen
        df_layout["contour number"].append(contour_number)
        df_layout["aspect ratio"].append(aspect_ratio)
        df_layout["extent"].append(extent)
        df_layout["Blue"].append(average_blue)
        df_layout["Green"].append(average_green)
        df_layout["Red"].append(average_red)
        df_layout["Hue"].append(average_hue)
        df_layout["class"].append(i)

        if count >= max_files:
            break
        count += 1


df = pd.DataFrame(df_layout)
# csv speichert weniger Metadaten, braucht weniger Speicher, sehr sinnvoll wenn Daten iwann sehr groß werden. "," als trennzeichen für ordentliche formatierung der anzeigetabelle
df.to_csv("output.csv", sep=",")

cv.waitKey(0)
cv.destroyAllWindows()
