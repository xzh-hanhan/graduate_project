# import numpy as np
# import pandas as pd
#
# data = pd.read_csv('D:\graduate_project\Emotion Recognition2.1\\fer2013\\fer2013\\fer2013.csv')
# num_of_instances = len(data) #获取数据集的数量
# print("数据集的数量为：",num_of_instances)
#
# pixels = data['pixels']
# emotions = data['emotion']
# usages = data['Usage']

# encoding:utf-8
# import pandas as pd
# import numpy as np
# import scipy.misc as sm
# import os
#
# from PIL import Image
#
# emotions = {
#     '0': 'anger',  # 生气
#     '1': 'disgust',  # 厌恶
#     '2': 'fear',  # 恐惧
#     '3': 'happy',  # 开心
#     '4': 'sad',  # 伤心
#     '5': 'surprised',  # 惊讶
#     '6': 'normal',  # 中性
# }
#
#
# # 创建文件夹
# def createDir(dir):
#     if os.path.exists(dir) is False:
#         os.makedirs(dir)
#
#
# def saveImageFromFer2013(file):
#     # 读取csv文件
#     faces_data = pd.read_csv(file)
#     imageCount = 0
#     # 遍历csv文件内容，并将图片数据按分类保存
#     for index in range(len(faces_data)):
#         # 解析每一行csv文件内容
#         emotion_data = faces_data.loc[index][0]
#         image_data = faces_data.loc[index][1]
#         usage_data = faces_data.loc[index][2]
#         # 将图片数据转换成48*48
#         data_array = list(map(float, image_data.split()))
#         data_array = np.asarray(data_array)
#         image = data_array.reshape(48, 48)
#
#         # 选择分类，并创建文件名
#         dirName = usage_data
#         emotionName = emotions[str(emotion_data)]
#
#         # 图片要保存的文件夹
#         imagePath = os.path.join(dirName, emotionName)
#
#         # 创建“用途文件夹”和“表情”文件夹
#         createDir(dirName)
#         createDir(imagePath)
#
#         # 图片文件名
#         imageName = os.path.join(imagePath, str(index) + '.jpg')
#         Image.fromarray(image).convert('L').save(imageName)
#         imageCount = index
#     print('总共有' + str(imageCount) + '张图片')
#
#
# if __name__ == '__main__':
#     saveImageFromFer2013('./fer2013/fer2013/fer2013.csv')
import cv2
from test_seg import delete_background,blend_images
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
path=delete_background(cv2.imread("./img_test/test2.jpg"))
path=blend_images(path, "./images_test/surprised.jpg")
img = mpimg.imread(path)

plt.figure(figsize=(10,10))
plt.savefig('test1.png')
plt.imshow(img)
plt.axis('off')
plt.show()