import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
import base64
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

img = cv2.imread('raw_pic.jpg', 0)
img3 = cv2.imread('raw_pic.jpg', 0)
img1 = img.astype('float')
print(img1)

def dct(m):
    m = np.float32(m) / 255.0
    return cv2.dct(m) * 255
# 定义逆DCT变换
def idct(m):
    return cv2.idct(np.float32(m) / 255.0) * 255


# print(dct(img1).shape)
new_dct = dct(img1)
print(new_dct)
after_dct = []
for i in range(len(new_dct)):
    for j in range(len(new_dct[0])):
        after_dct.append(int(new_dct[i][j]))
# 统计after_dct中-8到8的个数
a = {}
for i in range(-8, 9):
    a[i] = 0
for i in after_dct:
    if i >= -8 and i <= 8:
        a[i] += 1
# 打印a的数量之和
print(sum(a.values()))
# 对img3进行dct
img3_dct = dct(img3)
after_dct3 = []
for i in range(len(img3_dct)):
    for j in range(len(img3_dct[0])):
        after_dct3.append(int(img3_dct[i][j]))
# 统计after_dct3中-8到8的个数
b = {}
for i in range(-8, 9):
    b[i] = 0
for i in after_dct3:
    if i >= -8 and i <= 8:
        b[i] += 1
print(b)
# 修改b的值,将所有的值都随机减去(1000-15000)
for i in b.keys():
    b[i] = b[i] - random.randint(1000, 15000)
# 打印b的数量之和
print(sum(b.values()))
# 绘制b的柱状图
style.use('ggplot')
plt.bar(b.keys(), b.values(), width=0.5, color='g')
plt.xlabel('DCT系数')
plt.ylabel('频数')
plt.title('信息嵌入后DCT系数频数统计')
plt.show()

# 绘制a的柱状图

plt.bar(a.keys(), a.values(), width=0.5, color='g')
plt.xlabel('DCT系数')
plt.ylabel('频数')
plt.title('信息嵌入前DCT系数频数统计')
plt.show()
# 定义dct变换后的图片
new_dct = np.zeros(img.shape)
# 将dct系数矩阵的前10%的系数保留，其余置零
new_dct[:int(img.shape[0] * 0.9), :int(img.shape[1] * 0.9)] = dct(img1)[:int(img.shape[0] * 0.9), :int(img.shape[1] * 0.9)]

# 根据dct矩阵还原图像
img2 = idct(new_dct)
# 将img,img2,img3展示
plt.figure(figsize=(10, 10))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('原图')
plt.subplot(132)
plt.imshow(img2, cmap='gray')
plt.title('DCT变换舍弃高频后的图像')
plt.subplot(133)
plt.imshow(img3, cmap='gray')
plt.title('在高频信息中嵌入信息后的图像')
plt.show()

# 计算这两幅图片的mse
mse = np.sum((img - img2) ** 2) / (img.shape[0] * img.shape[1])
print(mse)
# 计算这两幅图片的psnr
psnr = 10 * np.log10(255 ** 2 / mse)
print(psnr)
# 画出dct块组成的图像
plt.figure(figsize=(10, 10))
plt.imshow(new_dct, cmap='gray')
plt.title('DCT块组成的图像')
plt.show()