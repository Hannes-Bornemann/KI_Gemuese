import numpy as np
import cv2

import feature_extracting_functions

image_path = "/home/michelle/Documents/Master_Sem1/Künstliche_Intelligenz/Projekt/onions/20231025_163732.jpg"
image = cv2.imread(image_path)

mask, contours = feature_extracting_functions.contour(image)

cv2.namedWindow("Maske", cv2.WINDOW_NORMAL)
cv2.imshow("Maske", mask)
cv2.resizeWindow("Maske", 800, 600)
#cv2.imwrite("Canny.png", edges)

box_points, extent, aspect_ratio = feature_extracting_functions.extent(image, contours)
histogramm = feature_extracting_functions.colour_hist(image, mask)

mean_r, mean_g, mean_b = feature_extracting_functions.mean_colours(image, mask)
print("Durchschnitt rot: ", mean_r, "Durchschnitt grün: ", mean_g, "Durchschnitt blau: ", mean_b)

cv2.waitKey(0)

