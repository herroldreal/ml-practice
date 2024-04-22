import cv2

recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read('trainer/trainer.yml')
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['None', 'John Doe', 'Susan Doe']

cam = cv2.VideoCapture(2)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    res, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        1.2,
        5,
        minSize=(int(minW), int(minH))
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 100:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('Camera => ', img)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

print("\n [INFO] Exiting program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
