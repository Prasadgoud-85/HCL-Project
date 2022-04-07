import numpy as np
import cv2

#web camera
cap=cv2.VideoCapture('video.mp4')

#Initialize substractor
algo=cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret,frame1=cap.read()
    grey=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(grey,(3,3),5)
    #applying on each frame
    img_sub=algo.apply(blur)
    dilat=cv2.dilate(img_sub,np.ones((5,5)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada=cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada=cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    countershape= cv2.findcontours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('Detecter',dilatada)
    # cv2.imshow('Video Orginal',frame1)

    if cv2.waitKey(1)==13:
        break

cv2.destroyAllWindows()
cap.release()