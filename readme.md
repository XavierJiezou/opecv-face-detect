【python】15行代码实现人脸检测（opencv）

----
# 1. 项目简介
利用opecv的python库，实现人脸检测。
# 2. 项目地址
> [https://github.com/XavierJiezou/opecv-face-detect](https://github.com/XavierJiezou/opecv-face-detect)
# 3. 依赖模块
```bash
pip install opencv-python
```
# 4. 完整代码
```python
import cv2

def face_detect(file_name, cascade_name):
    img = cv2.imread(file_name) # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 图片灰度化
    img_gray = cv2.equalizeHist(img_gray) # 直方图均衡化
    face_cascade = cv2.CascadeClassifier(cascade_name) # 加载级联分类器
    faces = face_cascade.detectMultiScale(img) # 多尺度检测
    for (x, y, w, h) in faces: # 遍历所有检测到的人脸
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 5) # 绘制矩形框
    cv2.imshow('Face detection', img) # 检测效果预览
    cv2.waitKey(0) # 保持窗口显示

if __name__ == "__main__":
    face_detect('img/test_1.jpg', 'haarcascade_frontalface_alt.xml')
```
# 5. 必要组件
**opencv**官方提供了**3**个已经训练好的人脸级联分类文件：
- `haarcascade_frontalface_alt.xml`：[点击下载](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml)

- `haarcascade_frontalface_alt2.xml`：[点击下载](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml)
- `haarcascade_frontalface_default.xml`：[点击下载](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
# 6. 成果展示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201225173149723.jpg#pic_center)
# 7. 对比分析
从**成果展示**图片来看，`alt`人脸检测的效果是最好的，`alt2`次之，`default`最差。
# 8. 引用参考
> [https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html)
> [https://docs.opencv.org/master/d2/d99/tutorial_js_face_detection.html](https://docs.opencv.org/master/d2/d99/tutorial_js_face_detection.html)