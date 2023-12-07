import cv2 as cv
import matplotlib.pyplot as plt


img = cv.imread('photos/Zwiebel/Zwiebel (2).jpg')
img = cv.resize(img, (800, 600))
# cv.imshow('Zwiebel (2)', img)

# plt.imshow(img) # zeigt Bild mit invertierten Farben mithilfe von matplotlib RGB BGR -> Rot und Blau vertauscht
# plt.show()

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#cv.imshow('HSV', hsv)

# BGR to L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# cv.imshow('HSV', lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) # opencv arbeitet mit BGR images, matplotlib mit RGB
cv.imshow('RGB', rgb)

plt.imshow(rgb)
plt.show()



cv.waitKey(0)
cv.destroyAllWindows()


