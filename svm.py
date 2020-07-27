import facenet_keras
import cv2
import numpy as np
from numpy import savez_compressed
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import pickle
from sklearn.metrics import accuracy_score

model = facenet_keras.facenet()

X = np.zeros((81,96,96,3))
Y_svm = np.zeros((81,1))
model.load_weights('weights.h5')
image_dir = "Final_faces"
model.summary()
count = 0
for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root, file)
        img = cv2.imread(path, 1)
        X[count ,:,:,:] = img
        count = count + 1
        print(count)
for i in range(9):
    for j in range(9):
        Y_svm[j + 9*i , 0] = i + 1
        print(i+1)
print(X.shape)
X_svm = model.predict(X)
X_train, X_test, y_train, y_test = train_test_split(X_svm, Y_svm, test_size=0.2)

print(X_svm.shape)
print(Y_svm.shape)
savez_compressed('face_detection.npz', X_train, y_train, X_test, y_test)

in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(X_train)
testX = in_encoder.transform(X_test)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(y_train)
trainy = out_encoder.transform(y_train)
testy = out_encoder.transform(y_test)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
#Saving Model
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))