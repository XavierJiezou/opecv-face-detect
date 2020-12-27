import matplotlib.pyplot as plt
import cv2
import os


class FaceDetector(object):
    def __init__(self):
        # if fname.endswith('.mp4'):
        #     self.video = cv2.VideoCapture(fname)
        # else:
        #     self.frame = cv2.imread(fname)
        target = input('请选择检测对象: 1 人 2 猫 3 动漫 (默认1) ')
        self.target = target if target else '1'

    @staticmethod
    def face_detect(frame, cascade_name):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_hist = cv2.equalizeHist(frame_gray)
        face_cascade = cv2.CascadeClassifier(cascade_name)
        faces = face_cascade.detectMultiScale(frame_hist)
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 5)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    def main(self):
        if self.target == '1':
            img_dir = 'img/human/'
            haar_dir = 'data/haarcascades/human/'
            lbp_dir = 'data/lbpcascades/human/'
            dst_dir = 'result/human/'
            haar_x=2
            haar_y=3
            lbp_x=1
            lbp_y=4
        elif self.target == '2':
            img_dir = 'img/cat/'
            haar_dir = 'data/haarcascades/cat/'
            lbp_dir = 'data/lbpcascades/cat/'
            dst_dir = 'result/cat/'
            haar_x = 1
            haar_y = 3
            lbp_x = 1
            lbp_y = 2
        elif self.target == '3':
            img_dir = 'img/anime/'
            haar_dir = None
            lbp_dir = 'data/lbpcascades/anime/'
            dst_dir = 'result/anime/'
            lbp_x = 1
            lbp_y = 2
        else:
            print('检测对象输入有误')
            os._exit(0)
        for item in os.listdir(img_dir):
            dst_path = dst_dir+item[:-4]
            os.makedirs(dst_path, exist_ok=True)
            def show(_dir,x,y,temp):
                dst_temp_dir = dst_path+temp
                os.makedirs(dst_temp_dir, exist_ok=True)
                frame = cv2.imread(img_dir+item)
                index = 1
                plt.figure(dpi=300)
                plt.subplot(x, y, index)
                plt.title('origin')
                save_name = dst_temp_dir+f'/{index}_origin.jpg'
                cv2.imwrite(save_name, frame)
                plt.imshow(plt.imread(img_dir+item))
                plt.axis('off')
                for each in os.listdir(_dir):
                    index+=1
                    cascade_name = _dir+each
                    result = self.face_detect(frame, cascade_name)
                    title = each.split('_')[-1][:-4]
                    save_name = dst_temp_dir+f'/{index}_{title}.jpg'
                    cv2.imwrite(save_name, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                    plt.subplot(x, y, index)
                    plt.title(title)
                    plt.imshow(result)
                    plt.axis('off')
                plt.savefig(f'{dst_path}{temp}.jpg', bbox_inches='tight')
                # plt.show()
            if haar_dir:
                show(haar_dir, haar_x, haar_y, '/haar')
            if lbp_dir:
                show(lbp_dir, lbp_x, lbp_y, '/lbp')


if __name__ == "__main__":
    FaceDetector().main()
