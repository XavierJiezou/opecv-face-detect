from face_detect import face_detect
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei'] 
import os


def main():
    img_dir = 'img'
    xml_dir = 'data/haarcascades'
    index = 1
    plt.figure(dpi=1200)
    for img in os.listdir(img_dir):
        img_name = os.path.join(img_dir, img)
        img_ori = plt.imread(img_name)
        plt.subplot(3, 4, index)
        plt.imshow(img_ori)
        plt.title('原始图像')
        plt.axis('off')
        index += 1
        for xml in os.listdir(xml_dir):
            xml_name = os.path.join(xml_dir, xml)
            img_det = face_detect(img_name, xml_name)
            plt.subplot(3, 4, index)
            plt.imshow(img_det)
            plt.title(f"{xml.split('_')[-1][:-4]}")
            plt.axis('off')
            index += 1
    plt.savefig('result.jpg', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
