from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
# 绘图中文
style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

"""
读取图片函数
"""
def read_image(image_path):
    img = Image.open(image_path)
    img = np.array(img)
    return img
"""
绘制图像像素直方图与原图  用子图表示
"""
def draw_hist(img):
    # 绘制图像像素直方图
    plt.subplot(1, 2, 1)
    plt.hist(img.ravel(), bins=256, density=1,  alpha=0.7)
    plt.xlabel(u"灰度值")
    plt.ylabel(u"频数")
    # 绘制原图
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.show()
"""
对比两张图的mse和psnr,ssim函数
"""
def compare_mse_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    psnr = 10 * np.log10(255.0 ** 2 / mse)

    return mse, psnr
"""
定义用DCT方法压缩图片
输入一张图片,返回压缩后的图片以及DCT块组成的图片
"""
def dct_compress(img):
    # 对图片进行灰度
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 对图片进行float32类型转换
    img = np.float32(img)
    # 对图片进行dct变换
    img_dct = cv2.dct(img)
    # 对图片进行逆dct变换
    img_idct = cv2.idct(img_dct)
    # 将图片转换为uint8类型
    img_idct = np.uint8(img_idct)
    # 返回压缩后的图片以及DCT块组成的图片
    return img_idct, img_dct
# 导入图片
img = read_image('raw_pic.jpg')
# 绘制图像像素直方图
draw_hist(img)
img1 = read_image('隐藏水印后的jphide.jpg')
draw_hist(img1)
img2 = read_image('嵌入宪法后的图片.jpg')
draw_hist(img2)
# 计算两张图的mse和psnr
mse, psnr = compare_mse_psnr(img, img1)
print('隐藏水印后的jphide.jpg的mse为：', mse)
print('隐藏水印后的jphide.jpg的psnr为：', psnr)
mse, psnr = compare_mse_psnr(img, img2)
print('嵌入宪法后的图片.jpg的mse为：', mse)
print('嵌入宪法后的图片.jpg的psnr为：', psnr)
# 对image1进行DCT变换
new_img1, new_dct1 = dct_compress(img1)
# 展示DCT变换后的图片和DCT块的图片
plt.subplot(1, 2, 1)
plt.imshow(new_img1, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(new_dct1, cmap='gray')
plt.show()

