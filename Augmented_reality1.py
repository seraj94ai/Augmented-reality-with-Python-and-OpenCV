
"""
An example of detecting ArUco markers with OpenCV.
"""

import cv2
import sys
import cv2.aruco as aruco


#device = 0 # Front camera
#try:
#    device = int(sys.argv[1]) # 0 for back camera
#except IndexError:
#    pass


cap = cv2.VideoCapture('2.mp4')
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()
 
    scale_percent = 30 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    # Check if frame is not empty
    if not ret:
        continue

    # Auto rotate camera
   # frame = cv2.autorotate(frame, device)

    # Convert from BGR to RGB
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    frame = aruco.drawDetectedMarkers(frame, corners, ids)
    print(corners)
    # Display the resulting frame
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything:
cap.release()
cv2.destroyAllWindows()
