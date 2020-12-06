import numpy as np
import argparse
import imutils
import time
import cv2
import os, errno
import urllib.request
from imutils.video import FPS

from vitis_ai_vart.facedetect import FaceDetect
import runner
import xir.graph
import pathlib
import xir.subgraph
from Spo2Calulation import face_detect_and_thresh,spartialAverage,MeanRGB,SPooEsitmate,preprocess

def get_subgraph (g):
    sub = []
    root = g.get_root_subgraph()
    sub = [ s for s in root.children
    if s.metadata.get_attr_str ("device") == "DPU"]
    return sub


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
    help = "input camera identifier (default = 0)")
ap.add_argument("-d", "--detthreshold", required=False,
    help = "face detector softmax threshold (default = 0.75)")
ap.add_argument("-n", "--nmsthreshold", required=False,
    help = "face detector NMS threshold (default = 0.55)")
args = vars(ap.parse_args())

if not args.get("input",False):
  inputId = 0
else:
  inputId = int(args["input"])
print('[INFO] input camera identifier = ',inputId)

if not args.get("detthreshold",False):
  detThreshold = 0.55
else:
  detThreshold = float(args["detthreshold"])
print('[INFO] face detector - softmax threshold = ',detThreshold)

if not args.get("nmsthreshold",False):
  nmsThreshold = 0.35
else:
  nmsThreshold = float(args["nmsthreshold"])
print('[INFO] face detector - NMS threshold = ',nmsThreshold)

# Initialize Vitis-AI/DPU based face detector
densebox_elf = "models/dpu_densebox.elf"
densebox_graph = xir.graph.Graph.deserialize(pathlib.Path(densebox_elf))
densebox_subgraphs = get_subgraph(densebox_graph)
assert len(densebox_subgraphs) == 1 # only one DPU kernel
densebox_dpu = runner.Runner(densebox_subgraphs[0],"run")
dpu_face_detector = FaceDetect(densebox_dpu,detThreshold,nmsThreshold)
dpu_face_detector.start()

# Initialize the camera input
print("[INFO] starting camera input ...")
cam = cv2.VideoCapture(inputId)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
if not (cam.isOpened()):
    print("[ERROR] Failed to open camera ", inputId )
    exit()
# url='http://192.168.1.104:8080/shot.jpg'

# start the FPS counter
fps = FPS().start()
frameCount = 0
totalFrame=250
final_sig=[]
# loop over the frames from the video stream
while True:
    
    # Capture image from camera
    ret,frame = cam.read()
    # imgResp = urllib.request.urlopen(url)
    # imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    # img = cv2.imdecode(imgNp,-1)
 
    # frame = imutils.resize(img,width=400)
    # # Vitis-AI/DPU based face detector
    faces = dpu_face_detector.process(frame)
    boxFrame = frame.copy()
        
    # loop over the faces
    # for i,(left,top,right,bottom) in enumerate(faces): 

    #     # draw a bounding box surrounding the object so we can
    #     # visualize it
    # cv2.rectangle( frame, (left,top), (right,bottom), (0,255,0), 2)
    try:
        (startX, startY, endX, endY) = faces[0].astype('int')
        startX2=int(int((endX+startX)/2)-50*1.3)
        endX2=int(int((endX+startX)/2)+50*1.3)
        startY2=int(int((endY+startY)/2)-50*1.3)
        endY2=int(int((endY+startY)/2)+50*1.3)
        cv2.rectangle(frame, (startX2, startY2), (endX2, endY2),(0, 0, 255), 2)

        boxFrame = boxFrame[startY2:endY2,startX2:endX2]
    
    except:
        boxFrame = boxFrame[int(100*0.9):int(200*1.1),int(150*0.9):int(250*1.1)]

    height, width, channels = boxFrame.shape
 
    if height>0 and width>0:
        lastfaceFrame = boxFrame
  
    else:
        boxFrame = lastfaceFrame
 
    if frameCount == 0:
        thresh,mask=face_detect_and_thresh(boxFrame)
        temp,min_value,max_value=spartialAverage(mask,boxFrame)
        final_sig.append(temp)
    
    elif frameCount<totalFrame and frameCount>1 :
        thresh,mask=face_detect_and_thresh(boxFrame)
        final_sig.append(MeanRGB(thresh,boxFrame,final_sig[-1],min_value,max_value))
    
    if frameCount==totalFrame:
        result = SPooEsitmate(final_sig,totalFrame,totalFrame,10)
        print(result)
        break
     # Display the processed image
    cv2.imshow("Face Detection", frame)
    frameCount = frameCount + 1
    
    key = cv2.waitKey(1) & 0xFF

    # Update the FPS counter
    fps.update()

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] elapsed FPS: {:.2f}".format(fps.fps()))

# Stop the face detector
dpu_face_detector.stop()
del densebox_dpu

# Cleanup
cv2.destroyAllWindows()
