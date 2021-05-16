import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2


NEW_SHAPE = (150, 150)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)

DEER_PATH = 'data/img/test/1/1_32446.jpg'
MODEL_PATH = 'data/checkpoint/model-12-0.0330.h5'

def get_prediction(model, new_shape, img):
    frame = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    frame = np.array([frame])
    pred = model.predict(x=frame)
    return pred


def add_dashboard(img, pred):
    if abs(pred - 1) < 0.01:
            back_colour = GREEN
            font_colour = MAGENTA
            message = 'deer'

    else:
        back_colour = MAGENTA
        font_colour = GREEN
        message = 'no deer'

    new_shape = (int(img.shape[1]*0.8), int(img.shape[0]*0.8))
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.rectangle(img,(25, 750),(1500, 850),back_colour, -1)

    message = message[:20] +" " + str(pred*100) + "%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, message, (50, 825), font, 2,font_colour, 6,cv2.LINE_AA)

    return img
    
if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture('data/video/deer0.mp4')
    pred = 0

    while(True):
        ret, img = cap.read()

        pred = get_prediction(model, NEW_SHAPE, img)[0][0]
        img = add_dashboard(img, pred)

        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows
