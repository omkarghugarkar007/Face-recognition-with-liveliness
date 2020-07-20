import cv2
from align import face_alignment

def face_detection_with_image(path):

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    img = cv2.imread(path)

    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)

    for x,y,w,h in faces:
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

    crop_img = img[y:y+h, x:x+w]
    resized = cv2.resize(crop_img, (int(img.shape[1]), int(img.shape[0])))

    cv2.imshow('Gray', resized)
    cv2.imwrite('align_with_face.png', resized)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

face_alignment()

face_detection_with_image("align.png")

