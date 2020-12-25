import cv2

class FaceDetector()


def face_detect(file_name, cascade_name):
    img = cv2.imread(file_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    face_cascade = cv2.CascadeClassifier(cascade_name)
    faces = face_cascade.detectMultiScale(img)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 5)
    # cv2.imwrite('result.jpg', img)
    cv2.imshow('Face detection', img)
    cv2.waitKey(0)
    # return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    face_detect('img/human/test_1.jpg', 'data/haarcascades/human/haarcascade_frontalface_alt.xml')
