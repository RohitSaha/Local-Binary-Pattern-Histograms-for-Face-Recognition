import cv2
import numpy as np

person = ['rohit', 'alaap', 'ansh']

model = cv2.createLBPHFaceRecognizer()
model.load("./trainer.xml")

face_cas = cv2.CascadeClassifier("./haarcascade_frontalface_alt_tree.xml")

capture = cv2.VideoCapture(0)

running = True
name = ""

while running:
    detect  = False
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray)
    #gray = cv2.flip(frame, 1)
    encode = [int(cv2.IMWRITE_JPEG_QUALITY), 150]
    result, imgencode = cv2.imencode('.jpg', gray, encode)
    data = np.array(imgencode)
    #LOAD_IMAGE_GRAYSCALE converts the image iinto a 2-D matrix of grayscale.
    decimg = cv2.imdecode(data, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    #test = decimg

    for (x,y,w,h) in faces:
        detect = True
        cv2.rectangle(frame, (x,y+10),(x+w,y+h+20),(255,0,0))
        test = decimg[y+10:y+h+20,x:x+w]
        confidence = 10000
        dim = (250 , 250)
        test = cv2.resize(test, dim, interpolation = cv2.INTER_AREA)

    if detect == True:
        try:
            num, confidence = model.predict(np.asarray(test, dtype=np.uint8))
            confidence = np.round(confidence)
        except:
            print "Error"
        name = person[num]
        cv2.putText(frame, name,(x,y),cv2.FONT_ITALIC,w*0.005,(255,255,255))

    cv2.imshow("Image", frame)
    if(cv2.waitKey(30)==27 & 0xff):
        running = False

cv2.destroyAllWindows()