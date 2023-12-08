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
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, file)
    img = cv.imread(input_path)
    img = cv.resize(img, (128, 128))
    cv.imwrite(output_path, img)

    ### Funktionen  ###
    average_blue, average_green, average_red, average_hue = feature_extraction_functions.mean_colours(img, excel_layout, count)

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
    if count >= 10:
        break

df = pd.DataFrame(excel_layout) 
df.to_excel('output.xlsx', index=False, startrow=0, startcol=0)

cv.waitKey(0)
cv.destroyAllWindows() 
