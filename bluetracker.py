import numpy as np
import cv2
from matplotlib import pyplot as plt
import time 

import socket
UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 25001
Message = "0"
clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

clientSock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
#lower = { 'blue':(100,100,100),'green':(40, 40, 100),'orange':(5, 40, 100),'red':(166, 84, 141) }
#upper = { 'blue':(140,255,255),'green':(60,255,255),'orange':(20,255,255),'red':(186,255,255)}

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (50,50)
fontScale              = 1
color              = (0,255,0)
thickness              = 2

img_width, img_height = 400,300
dim = (img_width, img_height)

lower_blue = np.array([100,100,60])
upper_blue = np.array([150,255,200])

lower_white = np.array([0,0,0], dtype=np.uint8)
upper_white = np.array([180,255,30], dtype=np.uint8)

lower_red = np.array([166, 84, 141], dtype=np.uint8)
upper_red = np.array([186,255,255], dtype=np.uint8)

lower_green = np.array([40, 40, 100], dtype=np.uint8)
upper_green = np.array([60,255,255], dtype=np.uint8)

lower_brown = np.array([10, 100, 20], dtype=np.uint8)
upper_brown = np.array([20, 255, 200], dtype=np.uint8)

cap = cv2.VideoCapture(0)
fps=cap.get(cv2.CAP_PROP_FPS)

n=0
TTime=0
result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         20, (640,480))
while(True):
    n+=1

    ret, frame = cap.read()
   #
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    blur = cv2.blur(hsv,(3,3))
    #blur = cv2.GaussianBlur(hsv,(2,2),3)
    #median = cv2.medianBlur(hsv,5)
    hsv=blur

    kernel = np.ones((5,5),np.uint8)
    #closing = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel)
    #hsv=closing


    mask = cv2.inRange(hsv, lower_brown, upper_brown)
#  mask2=cv2.inRange(hsv, lower_red, upper_red)
  #  mask = mask + mask2
    cv2.imshow('mask',cv2.resize(mask,(300,200)))
    

    contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
  #  contours2,hierarchy2 = cv2.findContours(mask2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    
    areas = [cv2.contourArea(c) for c in contours]
    arcp=areas.copy()
    #areas2 = [cv2.contourArea(c) for c in contours2]
    #print(areas)
    try:
        max_index = np.argmax(areas)
        arcp.pop(max_index)
       # print(areas)
       # max_index2 = np.argmax(areas2)

        cnt=contours[max_index]
        M = cv2.moments(cnt)

      #  cnt2=contours[max_index2]
      #  M2 = cv2.moments(cnt2)
       # print(M2)

        area = cv2.contourArea(cnt)
        print(area)
       # area2 = cv2.contourArea(cnt2)

        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    #
       # cx2 = int(M2['m10']/M2['m00'])
       # cy2 = int(M2['m01']/M2['m00'])
        #print(area)
       # print(cx,cy)
        #print("c va: ",cx2,cy2)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

       # x2,y2,w2,h2 = cv2.boundingRect(cnt2)
      #  cv2.rectangle(hsv,(x2,y2),(x2+w2,y2+h2),(0,255,0),2)

        Message1=str(-(cx-320)*(3.7/320))
        Message=str(round(float(Message1),2))
        cv2.putText(frame, "Values Sent: "+Message, (10,40), font,  1.2, (0,0,255), 2, cv2.LINE_AA) 
       # Message2=str(-(cx2-320)*(3.7/320))
       
        cv2.putText(frame, "Tracked Area: "+str(area), (10,80), font,  1, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Centroid: "+str(cx)+" "+str(cy), (10,120), font,  1, (0,0,0), 2, cv2.LINE_AA)  
        clientSock.sendto(bytes(Message,'utf-8'), (UDP_IP_ADDRESS, UDP_PORT_NO))
      
           # frame = cv2.putText(hsv, str(str(cx2)+","+str(Message2)), (cx2,cy2), font,  fontScale, color, thickness, cv2.LINE_AA) 
    except:
        clientSock.sendto(bytes(Message,'utf-8'), (UDP_IP_ADDRESS, UDP_PORT_NO))
        print("none exist")
        pass

    frame = cv2.putText(frame, "Tracking Brown", (cx,cy), font,  fontScale, 	(255,0,0), thickness, cv2.LINE_AA) 
    result.write(frame)
    cv2.imshow("frame",cv2.resize(frame,(300,200)))   
  
    k=cv2.waitKey(1)
    if(k==ord('q')):
        break
cap.release()
result.release()
cv2.destroyAllWindows()