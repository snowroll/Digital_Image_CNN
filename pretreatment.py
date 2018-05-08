# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt

I = open('lena.bmp')
gray = I.convert('L')
