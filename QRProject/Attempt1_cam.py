import os
import datetime
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode

with open('Team_Members.txt', 'r') as f:
    authorized_users = [l[:-1] for l in f.readlines() if len(l) > 2]

log_dir = 'Logged_Users.txt'
time_between_logs = 120

recent_access = {}

cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()
    qr_info = decode(frame)
    if len(qr_info) > 0:
        qr = qr_info[0]

        data = qr.data
        rect = qr.rect
        polygon = qr.polygon

        if data.decode() in authorized_users:
            cv2.putText(frame, f"Hello there, {data.decode()}", (rect.left, rect.top - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame = cv2.rectangle(frame, (rect.left, rect.top),
                                  (rect.left + rect.width, rect.top + rect.height), (0, 255, 0), 5)
            if data.decode() not in recent_access.keys() \
                    or time.time() - recent_access[data.decode()] > time_between_logs:
                recent_access[data.decode()] = time.time()
                with open(log_dir, 'a') as f:
                    f.write(f"{data.decode()},{datetime.datetime.now()}\n")
        else:
            cv2.putText(frame, "Nuh uh", (rect.left, rect.top - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            frame = cv2.rectangle(frame, (rect.left, rect.top),
                                  (rect.left + rect.width, rect.top + rect.height), (0, 0, 255), 5)

        frame = cv2.polylines(frame, [np.array(polygon)], True, (255,0,0), 5)
    cv2.imshow('Attendance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
