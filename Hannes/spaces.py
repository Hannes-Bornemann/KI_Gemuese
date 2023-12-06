import cv2 as cv

img = cv.imread('photos/Zwiebel/Zwiebel (2).jpg')
img = cv.resize(img, (800, 600))


cv.imshow('Zwiebel (2)', img)


cv.waitKey(0)
cv.destroyAllWindows()


