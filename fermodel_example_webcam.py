# This file allows to perform Emotion detection on frames grabbed from the webcam
# using OpenCV-Python

import cv2
from fermodel import FERModel
#from EmoPy.src.fermodel import FERModel

from datetime import datetime
import numpy as np

def tmp_log(array,now):
    log_array = np.empty((1,array.shape[0]+5))
    log_array[0,:array.shape[0]]=array
    log_array[0,array.shape[0]:]=\
            np.array([now.day,now.hour,now.minute,now.second,now.microsecond])
    return log_array

class ferClass:
    def __init__(self,file):
        self.target_emotions = ['calm', 'anger', 'happiness']
        self.model = FERModel(self.target_emotions, verbose=True)
        self.file = file
        self.mode = "no-print"

    def ret_fer(self):

        frame_string = self.model.predict(self.file,self.mode)
        data = np.array(frame_string)
        return data

def get_emotion_from_camera(face_oname,face_tmpname):

    # Specify the camera which you want to use. The default argument is '0'
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    video_capture.set(cv2.CAP_PROP_FPS, 15)

    ret = None

    file = 'image_data/image.jpg'
    emo = ferClass(file)

    face_stock = np.zeros((100,len(emo.target_emotions)+5))

    with open(face_oname,'a') as face_f:

      while(True):
        ret, frame = video_capture.read()
        cv2.imwrite(file, frame)
        if frame is not None:
            face_stock[:-1,:] = face_stock[1:]
            data=emo.ret_fer()
            print(data)

            now_time = datetime.now()
            face_stock[-1,:] = tmp_log(data,now_time)
            np.savetxt(face_tmpname,face_stock,fmt="%f",delimiter=",")
            np.savetxt(face_f,tmp_log(data,now_time),fmt="%f",delimiter=",")

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("Image could not be captured")


if __name__ == '__main__':
    oname = "test_face_log.csv"
    tmpname = "test_face.csv"
    get_emotion_from_camera(oname,tmpname)
