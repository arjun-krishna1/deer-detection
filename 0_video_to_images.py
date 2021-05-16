import numpy as np
import cv2

VIDEO_PATH = 'data_new/no_deer0.mp4'
IMAGE_BASE_PATH = 'data_new/no_deer0_img/no_deer0_'
NEW_SHAPE = (150, 150)

if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_PATH)

    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.resize(frame, NEW_SHAPE, interpolation=cv2.INTER_AREA)
        cv2.imshow('frame', frame)

        cv2.imwrite(IMAGE_BASE_PATH + str(counter) + '.jpg', frame)
        counter += 1

        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
