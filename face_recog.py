import numpy as np
import datetime
import telegram
import requests
import json
import cv2


# Bot telegram
token = ""
bot = telegram.Bot(token=token)
# URI request user_id
url = "https://api.telegram.org/bot" + token + "/getUpdates"
try:
    req = requests.get(url)
    user_id = json.loads(req.text)['result'][0]['message']['from']['id']
except ConnectionError:
    print("Lỗi kết nối!")
# Haar-cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("rtsp://192.168.3.16:8554/mjpeg/1")

# Vong lap phat hien khuon mat
while 1:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        image = cv2.imwrite("image.jpg", img)

        if (user_id is not None):
            bot.sendPhoto(chat_id = user_id, photo = open("image.jpg", "rb"), caption=datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y"))

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
