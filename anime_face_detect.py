import cv2

def face_detect(file_name, cascade_name):
    img = cv2.imread(file_name)  # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片灰度化
    img_gray = cv2.equalizeHist(img_gray)  # 直方图均衡化
    face_cascade = cv2.CascadeClassifier(cascade_name)  # 加载级联分类器
    faces = face_cascade.detectMultiScale(img)  # 多尺度检测
    for (x, y, w, h) in faces:  # 遍历所有检测到的动漫脸
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 5)  # 绘制矩形框
    cv2.imshow('Face detection', img)  # 检测效果预览
    cv2.waitKey(0)  # 保持窗口显示

if __name__ == "__main__":
    face_detect('img/anime/test_1.jpg', 'data/lbpcascades/anime/lbpcascade_animeface.xml')