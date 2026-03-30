import cv2
import numpy as np
import winsound
import threading

cap = cv2.VideoCapture(0) 
alarm_on = False   # alarm status

def play_alarm():
    print("ALARM FUNCTION CALLED")
    winsound.PlaySound('alarm.wav', winsound.SND_LOOP | winsound.SND_ASYNC)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_fire = np.array([0,120,120])
    upper_fire = np.array([50,255,255])

    mask = cv2.inRange(hsv, lower_fire, upper_fire)

    fire_pixels = cv2.countNonZero(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    fire_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 500:   # ignore small noise
            fire_detected = True
            x, y, w, h = cv2.boundingRect(cnt)

            
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

    cv2.imshow("Fire Mask", mask)

    if fire_pixels > 1500 or fire_detected:

        cv2.putText(frame,"FIRE DETECTED",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        if not alarm_on:
            alarm_on = True
            threading.Thread(target=play_alarm, daemon=True).start()

    else:
        alarm_on = False
        
        winsound.PlaySound(None, winsound.SND_PURGE)

    cv2.imshow("Fire Detection Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()