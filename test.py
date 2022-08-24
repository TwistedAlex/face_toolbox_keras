import PIL.Image
import os
import numpy as np


debg_path = "E:\\ResearchData\\MASKS\\s1p05\\test\\debg\\"
all_images = os.listdir(debg_path)
all_images.sort()
for image_name in all_images:
    img = np.asarray(PIL.Image.open(debg_path + image_name))
    print(img.shape)

debg_path = "E:\\ResearchData\\MASKS\\s1p05\\test\\orig\\"
all_images = os.listdir(debg_path)
all_images.sort()
for image_name in all_images:
    img = np.asarray(PIL.Image.open(debg_path + image_name))
    print(img.shape)