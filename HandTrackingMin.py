import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# class related to the hand detecting module
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # since Hands() parameters are already defined we can leave it empty
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cX, cY = int(lm.x*w), int(lm.y*h)
                print(id, cX, cY)
                if id == 4:
                    cv2.circle(img, (cX, cY), 15, (255, 255, 0), cv2.FILLED)
                if id == 8:
                    cv2.circle(img, (cX, cY), 15, (255, 255, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
