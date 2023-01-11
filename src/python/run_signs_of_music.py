import numpy as np
import pyautogui
import imutils
from mss import mss

import cv2
import copy
import scamp

from hand_detect import HandDetect

hand_detect = HandDetect()

scamp_session = scamp.Session()
piano = scamp_session.new_part("piano")

webcam: bool = True  # We can either get a live image from the webcam or use a single screenshot
if webcam:
    cap = cv2.VideoCapture(0)
else:  # Use an individual screenshot
    sct = mss()

with hand_detect.mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        ) as hands:
    while True:
        if webcam:
            ret, image = cap.read()
        else:
            ret = True
            mon = sct.monitors[0]
            image = np.array(sct.grab(mon))
            image = np.flip(image[:, :, :3], 2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not ret:  # Image was not successfully read!
            print('\rNo image!  Is the webcam available?', '', end='')
            continue

        raw_frame = copy.deepcopy(image)

        image = cv2.flip(image, 1)
        image_height, image_width, _ = image.shape

        for lm, mp_lm in hand_detect.detect_hand(hands=hands,
                                          image=raw_frame,
                                         ):
            hand_detect.mp_drawing.draw_landmarks(image, mp_lm, hand_detect.mp_hands.HAND_CONNECTIONS)

            print([int(-lm[8][1] * 20 + 56), int(-lm[8][1] * 20 + 56) + int((lm[4][1]-lm[8][1]) * 20)])
            piano.play_chord(
                #[int(-lm[8][1] * 20 + 56), int((lm[4][1]-lm[8][1])*40 + 56)],
                [int(-lm[8][1] * 20 + 56), int(-lm[8][1] * 20 + 56) + int((lm[4][1]-lm[8][1]) * 20)],
                lm[8][0],
                0.1)

        key = (cv2.waitKey(10) & 0xFF)

        image = cv2.resize(image, 
                            (int(image_width * 0.6), int(image_height * 0.6)), 
                            interpolation=cv2.INTER_AREA,
                            )

        cv2.imshow('frame', image)

        if key == ord('q'):
            break
        
if webcam:
    cap.release()
cv2.destroyAllWindows()
