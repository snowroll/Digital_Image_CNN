# -*- coding: utf-8 -*-
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal
import copy
import csv

#1. 图片大小一致　最好是正方形
#2. 8张图片转成csv
#3. label pixel..
#4. 标注　先空着

# 生成高斯算子的函数
def func(x, y, sigma = 1):
    return 100 *( 1 / (2 * np.pi * sigma)) * np.exp(-((x - 2)**2 + (y - 2)**2) / (2.0 * sigma ** 2))

# 生成标准差为5的5*5高斯算子
Gaussion = np.fromfunction(func,(5,5),sigma=5)

# Laplace扩展算子
Laplace_ex = np.array([[1, 1, 1],
                    [1,-8, 1],
                    [1, 1, 1]])

# 调整图片为正方形大小，方便后期处理 　***  文件路径file_path可自行添加修改  ***
file_path = "1.png"
Ori_img = Image.open(file_path)
(x, y) = Ori_img.size
#re_s = min(x, y)  *** re_s　调整图片大小尺寸，可修改  ***
re_s = 512
Re_img = Ori_img.resize((re_s, re_s), Image.ANTIALIAS)

#image = Image.open("lena.bmp").convert("L")
image = Re_img.convert("L")
image_array = np.array(image)

# 利用生成的高斯算子与原图像进行卷积对图像进行平滑处理
image_blur = signal.convolve2d(image_array, Gaussion, mode="same")
# 对平滑后的图像进行边缘检测
image2 = signal.convolve2d(image_blur, Laplace_ex, mode="same")
# 结果转化到0-255
image2 = ((image2 - image2.min()) / float(image2.max() - image2.min())) * 255

# 将大于灰度平均值的灰度值变成255（白色），便于观察边缘
image2[image2>image2.mean()] = 255
print(image2)

out = open('test.csv', 'a', newline = '')
csv_write = csv.writer(out, dialect = 'excel')

first_label = ['label']
pixel_num = re_s * re_s
for i in range(0, pixel_num):  #填写第一行
    label_str = 'pixel' + str(i)
    first_label.append(label_str)
csv_write.writerow(first_label)

tmp_image = image2 
for i in range(0, 4):  #旋转４次　对称４次　得８张测试图
    pixel_l = ['']
    pixel_r = ['']
    tmp_image_l = np.fliplr(tmp_image)
    tmp_image = np.rot90(tmp_image)
    #tmp_image = (image2/float(image2.max()))*255
    for j in range(0, pixel_num):
        pixel_l.append(tmp_image_l[(j // re_s), (j % re_s)])
        pixel_r.append(tmp_image[(j // re_s), (j % re_s)])
    csv_write.writerow(pixel_l)
    csv_write.writerow(pixel_r)
print("write over")

# 显示图像
plt.subplot(3,1,1)
plt.title('resize_img')
plt.imshow(image_array,cmap = cm.gray)
plt.axis("off")
plt.subplot(3,1,2)
plt.title('rotate_img')
plt.imshow(tmp_image,cmap = cm.gray)
plt.axis("off")
plt.show()