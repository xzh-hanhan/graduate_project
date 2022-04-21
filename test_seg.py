import paddlehub as hub
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np




def delete_background(image):
    human_seg = hub.Module(name="deeplabv3p_xception65_humanseg")
    result = human_seg.segmentation(images=[image],visualization=True)
    test_img_path = result[0]['save_path']
    return test_img_path

# img = mpimg.imread(test_img_path)
# plt.figure(figsize=(10,10))
# plt.imshow(img)
# plt.axis('off')
# plt.show()



def blend_images(fore_image, base_image):
    """
    将抠出的人物图像换背景
    fore_image: 前景图片，抠出的人物图片
    base_image: 背景图片
    """
    # 读入图片
    base_image = Image.open(base_image).convert('RGB')
    fore_image = Image.open(fore_image).resize(base_image.size)

    # 图片加权合成
    scope_map = np.array(fore_image)[:, :, -1] / 255
    scope_map = scope_map[:, :, np.newaxis]
    scope_map = np.repeat(scope_map, repeats=3, axis=2)
    res_image = np.multiply(scope_map, np.array(fore_image)[:, :, :3]) + np.multiply((1 - scope_map),
                                                                                     np.array(base_image))

    # 保存图片
    res_image = Image.fromarray(np.uint8(res_image))
    res_image.save("blend_res_img.jpg")
    return "./blend_res_img.jpg"

# test_img_path=delete_background(image_path="./img_test/happy1.jpeg")
# images1_path=blend_images(test_img_path, "./images_test/light.png")

# # 展示合成图片
# plt.figure(figsize=(10,10))
# img = mpimg.imread(images1_path)
# plt.imshow(img)
# plt.axis('off')
# plt.show()







