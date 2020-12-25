import cv2


def face_detect(img, cascade_name):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    face_cascade = cv2.CascadeClassifier(cascade_name)
    faces = face_cascade.detectMultiScale(img)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 5)
    cv2.imshow('Face detection', img)


def video_detect(file_name, cascade_name):
    video = cv2.VideoCapture(file_name) # 加载视频
    while True:
        success, img = video.read() # 读取视频帧
        if img is None:
            break
        else:
            face_detect(img, cascade_name) # 帧检测
        if cv2.waitKey(1) == 27:  # ESC退出
            break


if __name__ == "__main__":
    video_detect('video/test.mp4', 'data/haarcascades/haarcascade_frontalface_alt2.xml')
