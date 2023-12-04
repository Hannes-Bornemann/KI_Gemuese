import cv2 as cv

img =cv.imread('photos/Zwiebel/Zwiebel_1.jpg')
resized_img = cv.resize(img, (800, 600))

# cv.imshow('Zwiebel_1', resized_img)

# Convertig to grayscale
gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
# cv.imshow('Zwiebel_gray', gray)

# Blur
blur = cv.GaussianBlur(resized_img, (9,9), cv.BORDER_DEFAULT)
# cv.imshow('Zwiebel_blur', blur)

# Edge Cascade
canny = cv.Canny(resized_img, 125, 175)         # Kanten extrahieren
cv.imshow('Zwiebel_canny', canny)

# Dilating the image
dilated = cv.dilate(canny, (7,7), iterations=3) # Kanten aufdicken / verbinden
cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (7,7), iterations=3) # Kanten wieder verschmälern
cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_AREA)
cv.imshow('Resized', resized)

# Cropping
cropped = resized_img[50:200, 50:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)
cv.destroyAllWindows()