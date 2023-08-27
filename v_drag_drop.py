#%%
import numpy as np
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

# %%
cap = cv2.VideoCapture(0)
cap.set(3, 1200)
cap.set(4, 720)
detector = HandDetector(detectionCon = 0.8, maxHands=2)
colorR = 255,0,255

cx, cy, w, h = 100, 100, 120, 120

class DragRect():
    def __init__(self, center, size=[120, 120], color =[255,0,255] ):
        self.center = center
        self.size = size
        self.color = color
    
    def update(self, cursor):
        cx, cy = self.center
        w, h = self.size

        if cx-w//2 < cursor[0]<cx+w//2 and cy-h//2 <cursor[1]<cy+h//2:
            self.center = cursor
            self.color = [0,255,0]
        else:
            self.color = [255,0,255]

RectList=[]
for i in range(3):
    RectList.append( DragRect(center=[150*i+100, 100]) )  
alpha = 0.2
while True:
    success, img = cap.read()
    # img = cv2.flip(img, 1)
    hands, img= detector.findHands(img)
    # LmList, _ = detector.findPosition(img) # previous version
    if hands:
        lmList=hands[0]['lmList'] # List of 21 landmark points
        # print(lmList[8])
        bbox = hands[0]['bbox'] # bounding box info: x,y,w,h
        center = hands[0]['center'] # center coordinates of hand
        type  = hands[0]['type'] # hand type i.e. left or right
        if lmList:
            l,_ = detector.findDistance(lmList[4][:2], lmList[8][:2])
            num_up = detector.fingersUp(hands[0])
            # print(l)
            # print(sum(num_up))
            if l<40:
            # if sum(num_up)>2:
                cursor = lmList[8][:2]
                for rect in RectList:
                    rect.update(cursor)
                # if cx-w//2 < cursor[0]<cx+w//2 and cy-h//2 <cursor[1]<cy+h//2:
                #     colorR = 0, 255,0
                #     cx, cy = cursor
                # else:
                #     colorR = 255,0,255
    imgNew = np.zeros_like(img, np.uint8)
    for rect in RectList:        
        cx, cy = rect.center
        w, h = rect.size
        colorR = rect.color
        cv2.rectangle(imgNew, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx-w//2, cy-h//2, w, h), 20, rt=0)   
        
    out = img.copy()
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]


    cv2.imshow('CAM', out)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

