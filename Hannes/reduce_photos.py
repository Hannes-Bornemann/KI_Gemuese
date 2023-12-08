import cv2 as cv
import os

input_dir = 'photos/Zwiebel'
output_dir = 'photos_reduced/Zwiebel_reduced'

files = os.listdir(input_dir)   # speichert dateinamen in liste

for file in files:  # geht jedes element der liste durch
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, file)

    img = cv.imread(input_path)
    resized_img = cv.resize(img, (128, 128))
    cv.imwrite(output_path, resized_img)