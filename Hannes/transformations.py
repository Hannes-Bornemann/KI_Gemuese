import cv2 as cv
import numpy as np

img = cv.imread('photos/Zwiebel/Zwiebel (2).jpg')
img = cv.resize(img, (800, 600))

cv.imshow('Zwiebel (2)', img)

# Translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])   # eine list mit 2 lists drinnen, zuordnung x zu spalten und y zu reihen des bildes
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)
# -x --> left
# -y --> up
# x --> right
# y --> down
translated = translate (img, -100, 100)
cv.imshow('Translated', translated)

# Rotation
def rotate(img, angle, rotPoint=None):
    (height,width) = img.shape[:2] #setzt "hight" und "with" gleich der ersten zwei parameter der shape-liste, was der höher und breite entspricht

    if rotPoint is None:
        rotPoint = (width//2,height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated= rotate(img, -45)
cv.imshow('rotated', rotated)

rotated_rotated = rotate(rotated, -45)
cv.imshow('rotated_rotated', rotated_rotated)

# Flipping
flipped = cv.flip(img, 0) # 0=vertikaler flip, 1=horizontaler flip, -1=horizontal&vertikal
cv.imshow('flipped', flipped)

cv.waitKey(0)
cv.destroyAllWindows()