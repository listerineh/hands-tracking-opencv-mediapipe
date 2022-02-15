import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 640)

hands_class = mp.solutions.hands
hands = hands_class.Hands()

mpDraw = mp.solutions.drawing_utils


while True:
    positions = []

    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                width, heigth, c = img.shape
                corx, cory = int(lm.x * heigth), int(lm.y * width)
                positions.append([id, corx, cory])
                mpDraw.draw_landmarks(img, hand, hands_class.HAND_CONNECTIONS)

            if len(positions) == 21:
                thumb_x_max = max(positions[1][1], positions[2][1], positions[3][1], positions[4][1])
                thumb_x_min = min(positions[1][1], positions[2][1], positions[3][1], positions[4][1])
                thumb_y     = min(positions[1][2], positions[2][2], positions[3][2], positions[4][2])
                
                index_x_max = max(positions[5][1], positions[6][1], positions[7][1], positions[8][1])
                index_x_min = min(positions[5][1], positions[6][1], positions[7][1], positions[8][1])
                index_y     = min(positions[5][2], positions[6][2], positions[7][2], positions[8][2])
                
                middle_x_max = max(positions[9][1], positions[10][1], positions[11][1], positions[12][1])
                middle_x_min = min(positions[9][1], positions[10][1], positions[11][1], positions[12][1])
                middle_y     = min(positions[9][2], positions[10][2], positions[11][2], positions[12][2])
                
                ring_x_max = max(positions[13][1], positions[14][1], positions[15][1], positions[16][1])
                ring_x_min = min(positions[13][1], positions[14][1], positions[15][1], positions[16][1])
                ring_y     = min(positions[13][2], positions[14][2], positions[15][2], positions[16][2])
                
                little_x_max = max(positions[17][1], positions[18][1], positions[19][1], positions[20][1])
                little_x_min = min(positions[17][1], positions[18][1], positions[19][1], positions[20][1])
                little_y     = min(positions[17][2], positions[18][2], positions[19][2], positions[20][2])

                x1 = min(thumb_x_min, index_x_min, middle_x_min, ring_x_min, little_x_min) -30 # min x pos
                x2 = max(thumb_x_max, index_x_max, middle_x_max, ring_x_max, little_x_max) +30 # max x pos
                y1 = min(thumb_y, index_y, middle_y, ring_y, little_y)                     -30 # min y pos
                y2 = positions[0][2]                                                       +30 # max y pos (always 0)
                #img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                x_len = x2 - x1
                y_len = y2 - y1
                
                if x_len == y_len:
                    x1 = x1-2
                    y1 = y1-2
                    x2 = x2+2
                    y2 = y2+2
                elif x_len > y_len:
                    z = int((x_len - y_len) / 2)
                    x1 = x1-2
                    y1 = y1-z-2
                    x2 = x2+2
                    y2 = y2+z+2
                else:
                    z = int((y_len - x_len) / 2)
                    x1 = x1-z-2
                    y1 = y1-2
                    x2 = x2+z+2
                    y2 = y2+2
                
                if x1 > 20 and y1 > 20 and x2 < 920 and y2 < 520:
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                

    cv2.imshow('Lecture', img)

    if cv2.waitKey(1) == 27:
        break
