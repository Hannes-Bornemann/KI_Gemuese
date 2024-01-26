# import cv2
# import numpy as np
# import os
# import csv

# def extract_pixel_values(image_path):
#     image = cv2.imread(image_path)
#     return image.flatten()

# def process_images(folder_path, label):
#     data = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(folder_path, filename)
#             pixel_values = extract_pixel_values(image_path)
#             data.append([label] + list(pixel_values))

#     return data

# def main():
#     # Ordnerpfade für jedes Gemüse
#     potato_folder = "Michelle/photos_reduced/Kartoffel_reduced"
#     carrot_folder = "Michelle/photos_reduced/Karotte_reduced"
#     onion_folder = "Michelle/photos_reduced/Zwiebel_reduced"

#     # Daten für jedes Gemüse extrahieren
#     potato_data = process_images(potato_folder, 0)
#     carrot_data = process_images(carrot_folder, 1)
#     onion_data = process_images(onion_folder, 2)

#     # Alle Daten zusammenführen
#     all_data = potato_data + carrot_data + onion_data

#     # CSV-Datei schreiben
#     csv_filename = "gemuese_pixelwerte.csv"
#     with open(csv_filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerows(all_data)

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import os
import csv

def extract_pixel_values(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.flatten()

def process_images(folder_path, label):
    data = []
    pixel_count = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            pixel_values = extract_pixel_values(image_path)

            # Überprüfen, ob die Pixelanzahl bereits bekannt ist
            if pixel_count == 0:
                pixel_count = len(pixel_values)
            elif pixel_count != len(pixel_values):
                raise ValueError("Ungültige Pixelanzahl in Bildern")

            data.append([label] + list(pixel_values))

    return data

def main():
    # Ordnerpfade für jedes Gemüse
    potato_folder = "Michelle/photos_reduced/Kartoffel_reduced"
    carrot_folder = "Michelle/photos_reduced/Karotte_reduced"
    onion_folder = "Michelle/photos_reduced/Zwiebel_reduced"

    # Daten für jedes Gemüse extrahieren
    potato_data = process_images(potato_folder, 0)
    carrot_data = process_images(carrot_folder, 1)
    onion_data = process_images(onion_folder, 2)

    # Alle Daten zusammenführen
    all_data = potato_data + carrot_data + onion_data

    # CSV-Datei schreiben
    csv_filename = "gemuese_pixelwerte.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_data)

if __name__ == "__main__":
    main()

