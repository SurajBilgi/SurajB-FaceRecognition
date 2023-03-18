
import cv2,pickle
import os
from keras.models import Sequential,load_model
import keras.utils as image
import numpy as np


def face_recognizer():
    def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    def mask_validator(im,x,y,w,h):
        
            cv2.imwrite('temp.jpg',im)
            test_image=image.load_img('temp.jpg',target_size=(150,150,3))
            test_image=image.img_to_array(test_image)
            test_image=np.expand_dims(test_image,axis=0)
            pred=mymodel.predict(test_image)[0][0]
            pred=mymodel.predict(test_image)[0][0]
            if pred==1:
                # cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(im,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            else:
                # cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
                cv2.putText(im,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            return im

    # Creating patterns for face recognition
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    assure_path_exists("trainer/")

    # Loading the trained model
    recognizer.read('trainer/trainer.yml')
    mymodel=load_model('mymodel.h5')

    # Load prebuilt model for Frontal Face
    cascadePath = "haarcascade_frontalface_default.xml"

    # Creating Cascade classifier from prebuilt model
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize and start the video frame capture
    cam = cv2.VideoCapture(0)

    a_file = open("data.pkl", "rb")
    output = pickle.load(a_file)
    # print(output)
    a_file.close()

    name_list=list(output.keys())
    print(name_list)

    # Loop
    while True:
        # Read the video frame
        ret, im =cam.read()

        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        # Get all face from the video frame
        faces = faceCascade.detectMultiScale(gray, 1.2,5)
        
        # For each face in faces
        for(x,y,w,h) in faces:

            # Create rectangle around the face
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

            # Recognize the face belongs to which ID
            Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Check the ID if exist

            # num_lis = [1,2,3,4,5,6,7,8,9,17]
            num_lis = [i for i in range(1,30)]

            if(Id in num_lis and confidence > 30):
                Id = "{}".format(name_list[Id-1])+" {0:.2f}%".format(round(100 - confidence, 2))


            # Put text describe who is in the picture
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)

            im = mask_validator(im,x,y,w,h)

        # Display the video frame with the bounded rectangle
        cv2.imshow('im',im)

        # If 'q' is pressed, close program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Stop the camera
    cam.release()

    # Close all windows
    cv2.destroyAllWindows()
