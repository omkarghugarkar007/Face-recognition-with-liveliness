# Face-recognition-with-liveliness

Face.py is a separate file which uses face_recognition libraray

align.py contains the function to align the face using the dlib library

face_detection_with_provided_img uses align function to first align the face and then crop out the face using Haarcascade file. 

test.py file imports facenet() function from face_keras file which returns the model. The weights.h5 file contains the weights, thus completing the model. First the image is resized to (96,96) and is then feed to model. The output is a 128 dimensional vector.
The 128 dimensional outputs are feed to SVM for classification.

For the liveness, a input of 24 frames is feeded to a CNN which return the probabilty of being fake or not. It is working really good!

# Final file:

Video is captured using openCv and is feed to the CNN. It return a value for being spoof or not. If it's not a spoof, then the face is extracted , resized and feed to SVM for output.
