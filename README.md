# Face-recognition-with-liveliness

Face.py is a separate file which uses face_recognition libraray

align.py contains the function to align the face using the dlib library

face_detection_with_provided_img uses align function to first align the face and then crop out the face using Haarcascade file. 

test.py file import facenet() function from face_keras file which returns model. The weights.h5 file conatins the weights, thus completing the model. First the image is resized to (96,96) and is then feed to model. The output is a 128 dimensional vector.
