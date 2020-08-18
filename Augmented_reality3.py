"""
An example of using ArUco markers with OpenCV.
"""

import cv2
import sys
import cv2.aruco as aruco
import numpy as np


cap = cv2.VideoCapture('4.mp4')

im_src = cv2.imread('opencv_logo_with_text.png')

#def gen_video():    
#    cap = cv2.VideoCapture('chinookfinal.mp4')   
#    while(cap.isOpened()):
#        ret, frame = cap.read()
#        if not ret:
#            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#        yield (frame)


fn = 0  
cap_vid = cv2.VideoCapture('Amazing Nasa Videos Pexels · Free Stock Videos.mp4')
  
while cap.isOpened():
    fn += 1
    cap_vid.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret_v, im_src = cap_vid.read()
    # Capture frame-by-frame
    
    ret, frame = cap.read()
    scale_percent = 45 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100) 
    height = int(frame.shape[0] * scale_percent / 100) 
    dim = (width, height) 
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) 

    org_frame = frame
    # Check if frame is not empty
    if not ret_v:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Auto rotate camera
    #frame = cv2.autorotate(frame, device)

    # Convert from BGR to RGB
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #cv2.imshow('im_src', im_src)

    # resize image


    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if np.all(ids != None):
        for c in corners :
            x1 = (c[0][0][0], c[0][0][1]) 
            x2 = (c[0][1][0], c[0][1][1]) 
            x3 = (c[0][2][0], c[0][2][1]) 
            x4 = (c[0][3][0], c[0][3][1])   
            im_dst = frame
 
            size = im_src.shape
            pts_dst = np.array([x1, x2, x3, x4])
            pts_src = np.array(
                           [
                            [0,0],
                            [size[1] - 1, 0],
                            [size[1] - 1, size[0] -1],
                            [0, size[0] - 1 ]
                            ],dtype=float
                           );
            
            h, status = cv2.findHomography(pts_src, pts_dst)
             #project corners into frame
            temp = cv2.warpPerspective(im_src.copy(), h, (org_frame.shape[1], org_frame.shape[0])) 
            cv2.fillConvexPoly(org_frame, pts_dst.astype(int), 0, 16);
            org_frame = cv2.add(org_frame, temp)
        cv2.imshow('frame', org_frame)
       
    else:
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything:
cap.release()
cv2.destroyAllWindows()
