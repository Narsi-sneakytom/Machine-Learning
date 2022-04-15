import cv2 
import matplotlib.pyplot as plt  
import numpy as np
import time
from PIL import Image
import face_recognition

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def capture_face():
    img_name = "new_face.png".format(0)
    cv2.imwrite(img_name, frame) 
    print("{} written!".format(img_name))  
    return

def detect_face(image):
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        print("No faces found") 
    return image

cap=cv2.VideoCapture(0)
while 1: 

    ret,frame=cap.read() 
    frame=detect_face(frame) 
    cv2.imshow('Video face',frame)
    k=cv2.waitKey(1)
    if k==27: 
        break
    elif k==ord('s'):
        capture_face()
        break
cap.release() 
cv2.destroyAllWindows()

def detect_face_in_id(image):
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        print("No faces found") 
    for (x, y, w, h) in faces:
        x = x - 25
        y = y - 40
        image=cv2.rectangle(image, (x, y-5), (x + w + 50, y + h + 70), (27, 200, 10), 2)
    return image

def capture_face_id():
    img_name = "new_face_id.png".format(0)
    cv2.imwrite(img_name, frame) 
    print("{} written!".format(img_name))  
    return

cap=cv2.VideoCapture(0)
while 1: 

    ret,frame=cap.read(0) 
    frame=detect_face_in_id(frame) 
    cv2.imshow('id face',frame)
    k=cv2.waitKey(1)
    if k==27: 
        break
    elif k==ord('s'):
        capture_face_id()
        break
cap.release() 
cv2.destroyAllWindows()
im1=cv2.imread('new_face.png') 
im2=cv2.imread('new_face_id.png')

def crop_id(img): 
    face_img=img.copy() 
    face_rects=face_cascade.detectMultiScale(face_img,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30)) 
    for (x, y, w, h) in face_rects:
        x = x - 25
        y = y - 40
        roi_color = face_img[y-5:y + h + 70, x:x + w+50]
        print("[INFO] Object found. Saving locally.")
        cv2.imwrite('face_id_cropped.png', roi_color)
    return face_img

im2_1=crop_id(im2)

im3=cv2.imread('face_id_cropped.png')

def crop_face(img): 
    face_img=img.copy() 
    face_rects=face_cascade.detectMultiScale(face_img,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30)) 
    for (x, y, w, h) in face_rects:
        x = x - 25
        y = y - 40
        roi_color = face_img[y-5:y + h + 70, x:x + w+50]
        print("[INFO] Object found. Saving locally.")
        cv2.imwrite('face_cropped.png', roi_color)
    return face_img

p=crop_face(im1)

imm=cv2.imread('face_cropped.png')

t1=im3.shape

im4=Image.open('face_cropped.png')

resize=np.array(im4)

resize=im4.resize((t1[1],t1[0]),Image.ANTIALIAS)

resize=np.array(resize)

cv2.imwrite('face_cropped_final.png',resize)


im1 = face_recognition.load_image_file("C:/Users/Administrator/Desktop/test/face_cropped.png")

im2 = face_recognition.load_image_file("C:/Users/Administrator/Desktop/test/face_id_cropped.png") 
     
im1_encoding = face_recognition.face_encodings(im1)[0]

im2_encoding = face_recognition.face_encodings(im2)[0] 
     
results = face_recognition.compare_faces([im1_encoding], im2_encoding)

if results[0]:
    
    print('Identity matched !')

else:
    print('Pls show the correct identity !!!')
