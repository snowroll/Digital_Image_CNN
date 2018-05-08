# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal
import copy

#1. 图片大小一致　最好是正方形
#2. 8张图片转成csv
#3. label pixel..
#4. 标注　先空着

# 矩阵旋转函数
def rotate(matrix):
    # J = I
    # for i in range(len(J)-1):
    #     for j in range(len(J[i])):
    #         if j > i :
    #             tmp = J[i][j]
    #             J[i][j] = J[j][i]
    #             J[j][i] = tmp 
    # return J
    matrix[:] = map(list, zip(*matrix[::-1]))

# 生成高斯算子的函数
def func(x, y, sigma = 1):
    return 100 *( 1 / (2 * np.pi * sigma)) * np.exp(-((x - 2)**2 + (y - 2)**2) / (2.0 * sigma ** 2))

# 生成标准差为5的5*5高斯算子
Gaussion = np.fromfunction(func,(5,5),sigma=5)

# Laplace扩展算子
Laplace_ex = np.array([[1, 1, 1],
                    [1,-8, 1],
                    [1, 1, 1]])

# 调整图片为正方形大小，方便后期处理
Ori_img = Image.open("1.png")
(x, y) = Ori_img.size
re_s = min(x, y)
Re_img = Ori_img.resize((re_s, re_s), Image.ANTIALIAS)

#image = Image.open("lena.bmp").convert("L")
image = Re_img.convert("L")
image_array = np.array(image)

# 利用生成的高斯算子与原图像进行卷积对图像进行平滑处理
image_blur = signal.convolve2d(image_array, Gaussion, mode="same")

# 对平滑后的图像进行边缘检测
image2 = signal.convolve2d(image_blur, Laplace_ex, mode="same")

# 结果转化到0-255
image2 = (image2/float(image2.max()))*255

# 将大于灰度平均值的灰度值变成255（白色），便于观察边缘
image2[image2>image2.mean()] = 255

image_0_l = np.fliplr(image2)
image_90 = np.rot90(image2)  #防止赋值后由于指针的改变导致元素改变
image_90_l = np.fliplr(image_90)
image_180  = np.rot90(image_90)
image_180_l = np.fliplr(image_180)
image_270 = np.rot90(image_180)
image_270_l = np.fliplr(image_270)
# image_270 = copy.copy(rotate(image2))




# 显示图像
plt.subplot(3,1,1)
plt.imshow(image_array,cmap = cm.gray)
plt.axis("off")
plt.subplot(3,1,2)
plt.imshow(image_90_l,cmap = cm.gray)
plt.axis("off")
plt.subplot(3,1,3)
plt.imshow(image_90,cmap = cm.gray)
plt.axis("off")
plt.show()