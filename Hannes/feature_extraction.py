import cv2 as cv
import pandas as pd
import numpy as np
import os
import feature_extraction_functions

excel_layout = {
        'Blue': [],
        'Green': [],
        'Red': [],
        'Hue': [],
    }

input_dir = 'photos/Zwiebel'
output_dir = 'photos_reduced/Zwiebel_reduced'

files = os.listdir(input_dir)   # speichert dateinamen in liste

max_files = 5
count = 0
for file in files:
    # Bilder Verkleinern und in Ornder speichern
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, file)
    img = cv.imread(input_path)
    img = cv.resize(img, (400, 300))
    cv.imwrite(output_path, img)

    ### Funktionen  ###
    img, average_blue, average_green, average_red, average_hue = feature_extraction_functions.mean_colours(img, excel_layout, count)

    # Bild anzeigen lassen
    cv.imshow(f'img{count}', img)

    # werte an value im dictionary hängen
    value_bLue = excel_layout['Blue']
    value_bLue.append(average_blue)
    value_green = excel_layout['Green']
    value_green.append(average_green)
    value_red = excel_layout['Red']
    value_red.append(average_red)
    value_hue = excel_layout['Hue']
    value_hue.append(average_hue)

    count += 1
    if count >= max_files:
        break

df = pd.DataFrame(excel_layout) 
df.to_excel('output.xlsx', index=False, startrow=0, startcol=0)

cv.waitKey(0)
cv.destroyAllWindows() 
