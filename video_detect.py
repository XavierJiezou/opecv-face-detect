import cv2
from tqdm import tqdm


def face_detect(img, cascade_name):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    face_cascade = cv2.CascadeClassifier(cascade_name)
    faces = face_cascade.detectMultiScale(img)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 5)
    return img


def video_detect(file_name, cascade_name):
    video = cv2.VideoCapture(file_name)  # 加载视频
    fps = video.get(cv2.CAP_PROP_FPS)  # 帧率
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # 指定视频编码方式
    videoWriter = cv2.VideoWriter('result/test.mp4', 0x7634706d, fps, (w, h))  # 创建视频写对象
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    for i in tqdm(range(frame_count)):  # 帧数遍历
        success, img = video.read()  # 读取视频帧
        img = face_detect(img, cascade_name)  # 帧检测
        videoWriter.write(img)  # 视频对象写入


if __name__ == "__main__":
    video_detect('video/test.mp4', 'data/lbpcascades/anime/lbpcascade_animeface.xml')
