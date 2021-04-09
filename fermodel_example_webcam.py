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


def get_emotion_from_camera(face_oname):

    # Specify the camera which you want to use. The default argument is '0'
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    video_capture.set(cv2.CAP_PROP_FPS, 15)

    file = 'image_data/image.jpg'
    ret = None
    target_emotions = ['calm', 'anger', 'happiness']
    model = FERModel(target_emotions, verbose=True)

    mode = "no-print"

    with open(face_oname,'a') as face_f:

      while(True):
        ret, frame = video_capture.read()
        cv2.imwrite(file, frame)
        if frame is not None:
            # Can choose other target emotions from the emotion subset defined in
                    # fermodel.py in src directory. The function
            # defined as `def _check_emotion_set_is_supported(self):`

            frame_string = model.predict(file,mode)
            data = np.array(frame_string)
            np.savetxt(face_f,tmp_log(data,datetime.now()),fmt="%f",delimiter=",")
            print(target_emotions,np.array(frame_string))
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("Image could not be captured")


if __name__ == '__main__':
    get_emotion_from_camera("test_face.csv")
