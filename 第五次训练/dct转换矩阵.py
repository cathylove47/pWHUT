import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# 用PIL读取图
img = Image.open('raw_pic.jpg')
img = np.array(img)
print(img.shape)
# 将图像转换为灰度图
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 将图像转换为float32类型
img = np.float32(img)
print(img.shape)
# 对img进行dct变换
img_dct = cv2.dct(img)
print(img_dct.shape)
# 读取水印
watermark = Image.open('watermark.png')
watermark = np.array(watermark)
# 将水印转换为灰度图
watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
# 将水印转换为float32类型
watermark = np.float32(watermark)
# 将水印嵌入到img_dct中
img_dct[0:watermark.shape[0], 0:watermark.shape[1]] += watermark
# 对img_dct进行逆dct变换
img_idct = cv2.idct(img_dct)
# 将img_idct转换为uint8类型
img_idct = np.uint8(img_idct)
# 显示img_idct
plt.imshow(img_idct, cmap='gray')
plt.show()
# 保存img_idct
cv2.imwrite('img_idct.png', img_idct)
# 读取img_idct
img_idct = Image.open('img_idct.png')
img_idct = np.array(img_idct)
# 将img_idct转换为float32类型
img_idct = np.float32(img_idct)
# 将img_idct重新dct
img_idct_dct = cv2.dct(img_idct)
# 提取水印
extract_watermark = img_idct_dct[0:watermark.shape[0], 0:watermark.shape[1]] - img_dct[0:watermark.shape[0], 0:watermark.shape[1]]
# 显示提取的水印
plt.imshow(extract_watermark, cmap='gray')
plt.show()




