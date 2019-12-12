import cv2
from urllib import request
import numpy as np
import time
import random
font = cv2.FONT_HERSHEY_SIMPLEX

current_milli_time = lambda: int(round(time.time() * 1000))



stream = request.urlopen('http://192.168.000.000/stream')
#stream = cv2.VideoCapture(0)

bts = b''
count = 0
total_fails = 10
starttime = time.time()
t1 = current_milli_time()
t2 = t1
while True:
    bts += stream.read(1024)
    a = bts.find(b'\xff\xd8')
    b = bts.find(b'\xff\xd9')
    #print(a, b)

    if a != -1 and b != -1:
        jpg = bts[a:b+2]
        bts = bts[b+2:]
        img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        # get image height, width
        (h, w) = img.shape[:2
                            ]
        # calculate the center of the image
        center = (w / 2, h / 2)
        
        # angle90 = 90
        # angle180 = 180
        angle270 = 270
        
        scale = 1.5

        # Perform the counter clockwise rotation holding at the center
        # 90 degrees
        M = cv2.getRotationMatrix2D(center, angle270, scale)
        rotated270 = cv2.warpAffine(img, M, (h, w))
        
        # cv2.imshow('Video', img)

        # convert BGR image to a HSV image
        hsv = cv2.cvtColor(rotated270, cv2.COLOR_BGR2HSV) 

        # NumPy to create arrays to hold lower and upper range 
        # The “dtype = np.uint8” means that data type is an 8 bit integer [36, 28, 75]

        lower_range = np.array([136,87,111], dtype=np.uint8) 
        upper_range = np.array([180,255,255], dtype=np.uint8)      
        

        
        # create a mask for image
        mask = cv2.inRange(hsv, lower_range, upper_range)
        

        #Tracking the Red Color
        (contours, hierarchy)=cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        t1 = current_milli_time()
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if(area>300):
                
                x,y,w,h = cv2.boundingRect(contour)	
                rotated270 = cv2.rectangle(rotated270,(x,y),(x+w,y+h),(35,142,35),2)
                cv2.putText(rotated270,"FALHA",(x,y),cv2.FONT_HERSHEY_PLAIN, 0.7, (255,255,0))
                
            
                if count < total_fails and (t1-t2) >= 500: 
                    count += 1
                    print(count)
                    falhas = 'falha-%s.png' %(count) 
                    latitude = random.random() * random.randint(10,50)
                    longitude = random.random() * random.randint(10,50)
                    cv2.putText(rotated270,'Coordenadas',(1,30), font, 0.8,(255,255,255),1)
                    cv2.putText(rotated270,'X: -{}'.format(latitude),(1,45), font, 0.5,(255,255,255),1)
                    cv2.putText(rotated270,'Y: -{}'.format(longitude),(1,60), font, 0.5,(255,255,255),1)
                    cv2.imwrite(falhas, rotated270)
                    t2 = t1
                    

                
            
        # Bitwise-AND mask and original image
        cv2.imshow('Stream',rotated270)
        res = cv2.bitwise_and(rotated270,rotated270, mask= mask)
        #cv2.imshow('res',res)
        

        if cv2.waitKey(1) == 27:    
                break    