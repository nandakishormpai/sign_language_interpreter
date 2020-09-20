import tensorflow
import numpy as np
import cv2
import time
import os
from base_camera import BaseCamera

class Camera(BaseCamera):

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        np.set_printoptions(suppress=True)
        model = tensorflow.keras.models.load_model("primary_model.h5")

        #  0_Rock  1_Paper  2_Scissors  3_YourTurn

        s = ["images/0.png", "images/1.png", "images/2.png", "images/3.jfif"]

        # Setting default cam to webcam and necesseary variables
        img = cv2.VideoCapture(0)
        data = np.ndarray(shape=(1, 250, 250, 3), dtype=np.float32)
        firsttime = False
        exit = False


        CODES = {
            0: "nothing"
        }

        alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range (1,27):
            CODES[i]=alpha[i-1]

        CODES[27]="del"
        CODES[28]="space"

        start = time.time()
        end = time.time()
        final=""
        check = 0.0
        words=[]
        prev_move="del"
        while True:
            font = cv2.FONT_HERSHEY_SIMPLEX
            ret, frame = img.read()
            frame = cv2.flip(frame, 1)


            frame = cv2.rectangle(frame, (60, 100), (310, 350), (0, 0, 255), 3)
            frame2 = frame[100:350, 60:310]
            image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (50, 50))
            pred = model.predict(np.array([image]))
            move_code = CODES[np.argmax(pred[0])]
            gate = 1

            ret, frame = img.read()

            frame = cv2.flip(frame, 1)
            frame = cv2.rectangle(frame, (60, 100), (310, 350), (0, 0, 255), 3)
            cv2.putText(frame,  "Letter : {}".format(move_code), (63, 320),
            font, 1, (0, 0, 0), 2, cv2.LINE_AA)


            result = cv2.imread(s[3])
            if cv2.waitKey(1) & 0xff == ord('q'):
                exit = True
                break
            if(exit):
                break
            yield cv2.imencode('.jpg', frame)[1].tobytes()
