import cv2

cam = cv2.VideoCapture(2)
cam.set(3, 640)
cam.set(4, 480)

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


face_id = input('\n Enter user id and press <return> ==> ')
print("\n [INFO] Initializing face capture. Look the camera and wait...")

count = 0
while True:
    res, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
        cv2.imshow('Image', img)

    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break
    elif count >= 500:
        break

print("\n [INFO] Existing Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()