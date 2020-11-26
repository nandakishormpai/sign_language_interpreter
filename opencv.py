"""

Nanda Kishor M Pai

opencv.py : to predict sign language alphabets using the deep learning model we built

"""

import tensorflow
import numpy as np
from model import make_labels
import cv2

np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model("primary_model.h5")

def main():
    # Setting default cam to webcam and necesseary variables
    img = cv2.VideoCapture(0)
    data = np.ndarray(shape=(1, 250, 250, 3), dtype=np.float32)
    exit = False

    #labelling data
    CODES = {
        0: "nothing"
    }
    alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range (1,27):
        CODES[i]=alpha[i-1]

    CODES[27]="del"
    CODES[28]="space"

    while True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        ret, frame = img.read()
        frame = cv2.flip(frame, 1)

        #defining frame to be used
        frame = cv2.rectangle(frame, (60, 100), (310, 350), (0, 0, 255), 3)
        frame2 = frame[100:350, 60:310]
        image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (50, 50))
        pred = model.predict(np.array([image]))

        #predicting the letter
        move_code = CODES[np.argmax(pred[0])]
        window_width = 1200
        window_height = 820
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', window_width, window_height)

        ret, frame = img.read()

        frame = cv2.flip(frame, 1)
        frame = cv2.rectangle(frame, (60, 100), (310, 350), (0, 0, 255), 3)

        #displaying our prediction
        cv2.putText(frame,  "Letter : {}".format(move_code), (63, 320),
                font, 1, (0, 0, 0), 2, cv2.LINE_AA)

        if cv2.waitKey(1) & 0xff == ord('q'):
            exit = True
            break
        if(exit):
            break
        cv2.imshow('Frame', frame)

    img.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()

