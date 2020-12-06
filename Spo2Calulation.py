import cv2
import imutils
import numpy as np
import time
# from face_detection import FaceDetection
# from scipy import signal
import sys
from numpy.linalg import inv
# import dlib
import imutils
import time
import skin_detector

def face_detect_and_thresh(frame):
    skinM = skin_detector.process(frame)
    skin = cv2.bitwise_and(frame, frame, mask = skinM)
    # cv2.imshow("skin2",skin)
    # cv2.waitKey(1)
    return skin,skinM


def spartialAverage(thresh,frame):
    a=list(np.argwhere(thresh>0))
    # x=[i[0] for i in a]
    # y=[i[1] for i in a]
    # p=[x,y]
    if a:
        ind_img=(np.vstack((a)))
    else:
        return 0,0,0
    sig_fin=np.zeros([np.shape(ind_img)[0],3])
    test_fin=[]
    for i in range(np.shape(ind_img)[0]):
        sig_temp=frame[ind_img[i,0],ind_img[i,1],:]
        sig_temp = sig_temp.reshape((1, 3))
        if sig_temp.any()!=0:
            sig_fin=np.concatenate((sig_fin,sig_temp))
    # print(sig_fin)
    for _ in sig_fin:
        if sum(_)>0:
            # test_fin=np.concatenate((test_fin,_))
            test_fin.append(_)
    # print("min=>")
    a= [item for item in sig_fin if sum(item)>0]
    # print(min(a, key=sum))
    min_value=sum(min(a, key=sum))
    max_value=sum(max(a, key=sum))
    # print(sum1)
    img_rgb_mean=np.nanmean(test_fin,axis=0)
    # print(img_rgb_mean)
    return img_rgb_mean,min_value,max_value

def MeanRGB(thresh,frame,last_stage,min_value,max_value):
    # cv2.imshow("threshh",thresh)
    # print(thresh)
    # print("==<>>")
    # print(img_rgb)
    # cv2.waitKey(1)
    # print(img_rgb[0])
    # thresh=thresh.reshape((1,3))
    # img_rgb_mean=np.nanmean(thresh,axis=0)
    a= [item for item in frame[0] if (sum(item)>min_value and sum(item)<max_value)]
    # print(a)
    # a = filter(lambda (x,y,z) : i+j+k>764 ,frame[0])
    # print(a[1:10])
    # img_temp = [item for item in img_rgb if sum(item)>764]
    # print(frame[0])
    # print(img_temp)
    # print(np.mean(a, axis=(0)))
    if a:
        # print("==>")
        # print(a)
        # print("==>")
        img_mean=np.mean(a, axis=(0))
        # print(img_mean)

        return img_mean[::-1]
    else:
        return last_stage

def preprocess(z1,z2,detrended_RGB,window_size,size_video,duration,frame):
    temp=(int(size_video/duration))
    f=frame-2

    main_R=[]
    main_B=[]
    out=[]
    for i in range(len(detrended_RGB)-f+1):

        temp_R=z1[i:i+f-1]
        temp_B=z2[i:i+f-1]
        p=[list(a) for a in zip(temp_R, temp_B)]

        out.append(p)
        # if not main_R:
        #     main_R.append(temp_R)
        # else:
        #     main_R=[main_R,temp_R]
        #
        # if not main_B:
        #     main_B.append(temp_B)
        # else:
        #     main_B=[main_B,temp_B]


    # out=[main_R,main_B]
    # print(out[0])
    return out[0]


def SPooEsitmate(final_sig,video_size,frames,seconds):
    A = 100.6
    B = 4.834
    ten=10
    z1=[item[0] for item in final_sig]
    z3=[item[2] for item in final_sig]
    SPO_pre=[]
    for _ in range(len(z1)):
        SPO_pre.append([z1[_],z3[_]])
    Spo2 = preprocess(z1,z3,SPO_pre,ten,video_size,seconds,frames)

    R_temp = [item[0] for item in Spo2]
    DC_R_comp=np.mean(R_temp)
    AC_R_comp=np.std(R_temp)
    # print(DC_R_comp)
    # print(R_temp)
    # print(AC_R_comp)
    I_r=AC_R_comp/DC_R_comp

    B_temp = [item[1] for item in Spo2]
    DC_B_comp=np.mean(B_temp)
    AC_B_comp=np.std(B_temp)

    # print(I_r)
    I_b=AC_B_comp/DC_B_comp
    SpO2_value=(A-B*((I_b*650)/(I_r*950)))
    return SpO2_value