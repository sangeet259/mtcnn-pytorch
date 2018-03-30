import numpy as np
import cv2

from src import detect_faces, show_bboxes
from PIL import Image,ImageShow

import time
import os

cap = cv2.VideoCapture(0)

while(True):
    try:


        # Capture frame-by-frame
        _, cv2_im = cap.read()

        cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        bounding_boxes, landmarks = detect_faces(pil_im)


        pil_result = show_bboxes(pil_im, bounding_boxes, landmarks)
        opencvImage = np.array(pil_result)
        opencvImage = cv2.cvtColor(np.array(pil_result), cv2.COLOR_RGB2BGR)
        cv2.imshow('frame',opencvImage)
    except:
        pass



    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
