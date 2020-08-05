import cv2
import pytesseract as tess
from PIL import Image
import dlib
img=cv2.imread('down.jpg')
real=cv2.imread('download.jpg' , 0)
Falseimg = cv2.imread('down.jpg', 0)
Falseimg=cv2.resize(Falseimg,(200,200))
real=cv2.resize(real,(200,200))
ret,thresh1 = cv2.threshold(Falseimg,75,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(real,127,255,cv2.THRESH_BINARY)
cv2.imshow('Bina1',thresh2)
cv2.imshow('Bina0',thresh1)
#cv2.imshow('Real',real)
#cv2.imshow('False',Falseimg)
cv2.waitKey(0)

app=dlib.get_frontal_face_detector()
cv2.destroyAllWindows()