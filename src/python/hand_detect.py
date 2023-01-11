import mediapipe as mp
import cv2
import time


class HandDetect():
    """
    Responsible for detecting hand landmarks.
    """

    def __init__(self, detect_threshold=0.90):
        self.detect_threshold = detect_threshold
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        self.landmarks = [x.name for x in mp.solutions.hands.HandLandmark]

    def image_preprocessing(self, image):
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        return image

    def detect_hand(self, hands, image):
        """
        Preprocess iamge and identify hand in image.
        
        Return: Hand landmarks.
        """

        image = self.image_preprocessing(image)
        mp_results = hands.process(image)

        if mp_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(mp_results.multi_hand_landmarks,
                                                  mp_results.multi_handedness):

                if handedness.classification[0].score <= self.detect_threshold:
                    continue

                landmarks = []
                for lm in self.landmarks:

                    landmark_idx = self.mp_hands.HandLandmark[lm]

                    landmarks.append((hand_landmarks.landmark[landmark_idx].x,
                                      hand_landmarks.landmark[landmark_idx].y,
                                      hand_landmarks.landmark[landmark_idx].z,
                                    ))

                yield landmarks, hand_landmarks
        else:
            pass
            # print("No hands found.", "", end="")
