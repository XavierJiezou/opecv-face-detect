【python】15行代码实现人脸检测（opencv）

----
# 1. 项目简介
利用**opecv**的**python**库及训练好的**级联分类器**实现人脸检测。
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
    face_detect('test.jpg', 'haarcascade_frontalface_alt.xml')
```
# 5. 必要组件
**opencv**官方提供了**8**个已经训练好的人脸级联分类文件：
## 5.1. haar级联特征分类器（精度高）
- `haarcascade_frontalface_default.xml`：[点击下载](https://cdn.jsdelivr.net/gh/XavierJiezou/opecv-face-detect@master/data/haarcascades/haarcascade_frontalface_default.xml)
- `haarcascade_frontalface_alt.xml`：[点击下载](https://cdn.jsdelivr.net/gh/XavierJiezou/opecv-face-detect@master/data/haarcascades/haarcascade_frontalface_alt.xml)

- `haarcascade_frontalface_alt2.xml`：[点击下载](https://cdn.jsdelivr.net/gh/XavierJiezou/opecv-face-detect@master/data/haarcascades/haarcascade_frontalface_alt2.xml)
- `haarcascade_frontalface_alt_tree.xml`：[点击下载](https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/haarcascades/haarcascade_frontalface_alt_tree.xml)
- `haarcascade_profileface.xml`：[点击下载](https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/haarcascades/haarcascade_profileface.xml)
## 5.2. lbp级联特征分类器（速度快）
- `lbpcascade_frontalface.xml`：[点击下载](https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/lbpcascades/lbpcascade_frontalface.xml)
- `lbpcascade_frontalface_improved.xml`：[点击下载](https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/lbpcascades/lbpcascade_frontalface_improved.xml)
- `lbpcascade_profileface.xml`：[点击下载](https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/lbpcascades/lbpcascade_profileface.xml)
----
`frontalface`对正脸检测效果好，`profileface`专门针对侧脸进行检测。一般来说，`haar`特征检测精度更高，而`lbp`特征检测用时更短。
# 6. 成果展示
## 6.1. 测试样例1
- **haar**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228095556968.jpg?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTUxNTYw,size_16,color_FFFFFF,t_70#pic_center)
- **lbp**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228095608844.jpg?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTUxNTYw,size_16,color_FFFFFF,t_70#pic_center)
##  6.2. 测试样例2
- **haar**

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020122809565139.jpg?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTUxNTYw,size_16,color_FFFFFF,t_70#pic_center)
- **lbp**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228095701851.jpg?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTUxNTYw,size_16,color_FFFFFF,t_70#pic_center)

## 6.3. 测试样例3
- **haar**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228095717575.jpg?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTUxNTYw,size_16,color_FFFFFF,t_70#pic_center)
- **lbp**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228095725904.jpg?x-oss-process=image,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTUxNTYw,size_16,color_FFFFFF,t_70#pic_center)

# 7. 对比分析
从测试结果来看，`haar`级联特征的`alt`人脸检测的精度是最高的，其次就是`alt2`，但如果对精度要求不高，可以采用`lbp`级联特征检测，因为这个花费的时间很短。
# 8. 引用参考
> [https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html)
> [https://docs.opencv.org/master/d2/d99/tutorial_js_face_detection.html](https://docs.opencv.org/master/d2/d99/tutorial_js_face_detection.html)

# 9. 相关推荐
> [【python】15行代码实现猫脸检测（opencv）](https://blog.csdn.net/qq_42951560/article/details/111831532)

![](https://img-blog.csdnimg.cn/20201228102022683.jpg#pic_center)
> [【python】15行代码实现动漫脸检测（opencv）](https://blog.csdn.net/qq_42951560/article/details/111831797)

![](https://img-blog.csdnimg.cn/20201228103025477.jpg#pic_center)