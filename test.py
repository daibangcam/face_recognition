from PIL import Image
import numpy as np
import cv2
import os
# ################################
# name='Face-Recognition/dataSet/face-1.2.jpg'
# print(os.path.split(name))
# print(os.path.split(name)[-1])
# print(os.path.split(name)[1].split('.'))
# print(os.path.split(name)[1].split('.')[1])
##################################
def face_recognition(img):
    # chuyển ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # tìm khuôn mặt trong ảnh
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    # print("Found " + str(len(faces)) + " face(s)")
    # quét lần lượt tất cả khuôn mặt trong ảnh
    for (x, y, w, h) in faces:
        # so sánh mặt trong model
        id_predicted, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # kiểm tra confidence < 100 : "0" là kết quả perfect
        if confidence < 90:
            id_predicted = "Known"
            cv2.rectangle(img, (x - 10, y - 20), (x + w + 20, y + h + 40), (0, 255, 0), 2)
            cv2.putText(img, str(id_predicted), (x, y + h + 30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
            stt = 1
        else:
            id_predicted = "Unknown"
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, str(id_predicted), (x, y + h + 30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
            stt = 0
    return stt
################################
# load cascade để nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
################################
# khai báo folder chứa model huấn luyện
train_path = 'trainer'
train_list = os.listdir(train_path)
print(train_list)
recognizer = cv2.face_LBPHFaceRecognizer.create()
# nhận diện ảnh
img = cv2.imread('test.jpg')
for i in train_list:
    print('using ' + os.path.split(f'{train_path}/{i}')[-1])
    recognizer.read(f'{train_path}/{i}')
    status = face_recognition(img)
    if status == 1:
        break
print('model ' + os.path.split(f'{train_path}/{i}')[-1] + ' known this person')
