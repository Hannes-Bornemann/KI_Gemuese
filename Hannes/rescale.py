import cv2 as cv

img =cv.imread('Zwiebel (2).jpg')
# cv.namedWindow('kaninchen1', cv.WINDOW_NORMAL)  #Passt anzeigegröße an bildschirmgröße des laptops an (FUNKTIONIERT NOCH NICHT!!)
# cv.resizeWindow('kaninchen1', 800, 600) # -> Passt nur Anzeigegröße des Bildes an ohne Pixelanzahl zu verändern (FUNKTIONIERT NOCH NICHT!!)
# cv.imshow('kaninchen1', img)

def rescaleFrame(frame, scale=0.2):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA) # INTER_AREA for downsizing, INTER_LINEAR or INTER_CUBIC for upsizing


resized_image = rescaleFrame(img)
cv.imshow('popupwindowname', resized_image)

cv.waitKey(0)
