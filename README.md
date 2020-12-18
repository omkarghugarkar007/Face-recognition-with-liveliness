# Face-recognition-with-liveliness

### Facenet

align.py contains the function to align the face using the dlib library

face_detection_with_provided_img uses align function to first align the face and then crop out the face using Haarcascade file. 

test.py file imports facenet() function from face_keras file which returns the model. The weights.h5 file contains the weights, thus completing the model. First the image is resized to (96,96) and is then feed to model. The output is a 128 dimensional vector.
The 128 dimensional outputs are feed to SVM for classification.

For the liveness, a input of 24 frames is feeded to a CNN which return the probabilty of being fake or not. It is working really good!

'''
Final.py:
'''

Video is captured using openCv and is feed to the CNN. It return a value for being spoof or not. If it's not a spoof, then the face is extracted , resized and feed to SVM for output.

#### System Description:

![image](https://user-images.githubusercontent.com/62425457/102617694-9b694c80-415f-11eb-8cf1-8e5f9de1a3e2.png)

#### An Example

![image](https://user-images.githubusercontent.com/62425457/102618011-0dda2c80-4160-11eb-88f1-8c59cd413ad9.png)

Link to Documentation - https://docs.google.com/document/d/1MXrBg9g6HKCP9ZonT-44aQq2yynLJuYdaVX7vL3jm8I/edit?usp=sharing

### Face Recognition Library

It contains the Code to recognie face using the Face Recognition Library. 

