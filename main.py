from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import math
import cvzone
from datetime import datetime

def find_h():
    h, m, s = str(datetime.now()).split(sep=":")
    currh = float(h) + float(m/60) + float(s/360)
    return currh

def find_speed(id, frame_rate):
    seconds = (frame_count - s_start[id]) / 60 #considering 60 frames per second
    speed = (30/seconds)*360/1000 #assuming 30m distance from line1 to line2
    del s_start[id]
    return round(speed, 2)



model = YOLO('yolov8l.pt')

v_path = "video.mp4"
vcap = cv2.VideoCapture(v_path)
tracker = DeepSort(max_age=20)

ret = True
ids =[]
nid = []
carid = []
mcycid = []
busid = []
truckid = []
s_start = {}
speed = {}

colorR = {2:(235, 131, 5),
          3:(5, 131, 235),
          5:(131, 235, 5),
          6:(131, 235, 5),
          7:(131, 235, 5)}

frame_count = 0

frame_w = int(vcap.get(3))
frame_h = int(vcap.get(4))
size = (frame_w, frame_h) 

video_result = cv2.VideoWriter('result.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 10, size) 

while ret:

    ret, frame = vcap.read()
    frame_count = frame_count + 1
    dit = []
    results = model.track(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0]
            x1, y1, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
            cf = math.ceil((b.conf[0]*100))/100
            cls = int(b.cls[0])
            if cls == 2 or 3 or 5 or 6 or 7:
                carr = ([x1, y1, w, h], cf, cls)
                dit.append(carr)
            # cvzone.cornerRect(frame, (x1, y1, w, h), rt=2, colorR=(0,255,0))


    print(dit)
    trackResult = tracker.update_tracks(dit, frame=frame)
    for r in trackResult:

        print(r.to_tlbr())
        x1, y1, x2, y2 = r.to_tlbr()
        x1, y1, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
        id = r.track_id
        cls = r.det_class
        
        cvzone.cornerRect(frame, (x1, y1, w, h), l=5, t=2, rt=1, colorR=colorR[cls if cls else 2], colorC=(255, 255, 0))
        cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(40, y1)), scale=1, thickness=2, offset=1, colorR=(74, 84, 5))
        cv2.line(frame, (256, 256), (720, 256), (100, 230, 100), thickness=4)
        cv2.line(frame, (226, 326), (740, 326), (100, 230, 100), thickness=4)
        cv2.line(frame, (960, 280), (1200, 460), (100, 230, 100), thickness=4)
        mx = x1 + w/2
        my = y1+ h/2
        if (256<=mx<=720 and 200<=my<=320) or 940<=mx<=1240 and 220<=my<=500:
            if nid.count(id) == 0:
                nid.append(id)
                if cls == 2:
                    carid.append(id)
                elif cls == 3:
                    mcycid.append(id)
                elif cls == 5 or 7:
                    busid.append(id)
        
        if (256<=mx<=720 and 250<=my<=260): #speed line enter
            if not id in s_start:
                s_start.update({id: frame_count})
        
        if (226<=mx<=740 and 320<=my<=330): #speed line exit
            if id in s_start:
                print(s_start)
                speed[id] = find_speed(id, frame_count)

        if id in speed:
            cvzone.putTextRect(frame, f'speed: {speed[id]}km/hr', (x1+50, y1), scale=1, thickness=2, offset=1, colorR=(74, 84, 5))

    cvzone.putTextRect(frame, f'Cars: {int(len(carid))} MotorCycles: {int(len(mcycid))} Large vehicles: {int(len(busid))} ', (10, 32), font=cv2.FONT_HERSHEY_DUPLEX, scale=1, thickness=2, offset=5, colorR=(59, 51, 26))

    video_result.write(frame)
    # cv2.imshow('frame', frame)
    cv2.waitKey(1)

vcap.release()
video_result.release() 
    
# Closes all the frames 
cv2.destroyAllWindows() 