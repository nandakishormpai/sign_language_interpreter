import tensorflow
import numpy as np
import cv2
import time

np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model("model.h5")

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

while True:
    font = cv2.FONT_HERSHEY_SIMPLEX
    ret, frame = img.read()
    frame = cv2.flip(frame, 1)


    frame = cv2.rectangle(frame, (320, 100), (570, 350), (0, 0, 255), 3)
    frame2 = frame[100:350, 320:570]
    image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (50, 50))
    pred = model.predict(np.array([image]))
    #image_array = np.asarray(frame2)
    #normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    #data[0] = normalized_image_array
    #pred = model.predict(data)


    move_code = CODES[np.argmax(pred[0])]

    start = time.time()
    end = time.time()
    check = 0.0
    gate = 1
    window_width = 1200
    window_height = 820
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', window_width, window_height)

    ret, frame = img.read()

    frame = cv2.flip(frame, 1)
    frame = cv2.rectangle(frame, (320, 100), (570, 350), (0, 0, 255), 3)
    cv2.putText(frame,  "----------", (3, 87),
            font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame,  "Letter : {}".format(move_code), (25, 117),
            font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame,  "----------", (3, 187),
            font, 1, (0, 0, 0), 2, cv2.LINE_AA)





    result = cv2.imread(s[3])
    if cv2.waitKey(1) & 0xff == ord('q'):
        exit = True
        break
    if(exit):
        break
    cv2.imshow('Frame', frame)



img.release()
cv2.destroyAllWindows()