import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 4

imgdir = 'photos'
images =[]
classNames = [] #image names that shoul be displayed
nameList = os.listdir(imgdir)
print(nameList) 

for cls in nameList:
    currentImg = cv2.imread(f'{imgdir}/{cls}')
    images.append(currentImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

#encode the images

def FindEncodings(images):
    encodeList = []
    for img in images:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)
    return encodeList

def MarkAttendance(name):
    with open('F:/face recognition/attendance.csv','r+') as f:
        Data = f.readlines()
        nameList = []
        for lines in Data:
            entry = lines.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            f.seek(0, 2)
            f.writelines(f'\n{name},{time}')

KnownEncodeList = FindEncodings(images)
print('Encoding Successful')



cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    success, image = cap.read()
    imageS = cv2.resize(image,(0,0),fx=0.25, fy=0.25)
    imageS = cv2.cvtColor(imageS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imageS)
    encodeCurFrame = face_recognition.face_encodings(imageS,facesCurFrame)
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matchings = face_recognition.compare_faces(KnownEncodeList, encodeFace)
        faceDis = face_recognition.face_distance(KnownEncodeList,encodeFace)
        print(faceDis)
        
        matchingItem = np.argmin(faceDis)
        if matchings[matchingItem]:
            Name = classNames[matchingItem]
            print(Name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(image,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(image, Name,(x1-6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            MarkAttendance(Name)
            
            
    cv2.imshow('Webcam', image)
    cv2.waitKey(1)



