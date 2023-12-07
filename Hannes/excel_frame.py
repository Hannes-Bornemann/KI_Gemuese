import cv2 as cv
import numpy as np
import pandas as pd

excel_layout = {
        'Blue': [],
        'Green': [],
        'Red': [],
        'Hue': [],
        #hier weitere Merkmale hin
    }

for i in range (1,2):

    path = f'photos/Zwiebel/Zwiebel ({i}).jpg'
    img = cv.imread(path)
    img = cv.resize(img, (800, 600))


    # hier Merkmale einfügen 
   
    # und in dataframe schreiben:
    # value_hue = excel_layout['Hue']
    # value_hue.append(average_hue)

    
    df = pd.DataFrame(excel_layout) 
    df.to_excel('output.xlsx', index=False, startrow=0, startcol=0)

cv.waitKey(0)
cv.destroyAllWindows() 