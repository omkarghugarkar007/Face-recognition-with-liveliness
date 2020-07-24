from model import create_model
import facenet_keras
import cv2
import numpy as np
model = facenet_keras.facenet()

model.load_weights('weights.h5')

#model.summary()

path = "D:/Face Recognition with lively detection/Trial/biden.jpg"
img = cv2.imread(path, 1)
img = cv2.resize(img, (96,96))
x_train = np.array([img])
y = model.predict(x_train)

print(y)