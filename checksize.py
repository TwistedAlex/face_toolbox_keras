import PIL.Image
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


path = "E:\\ResearchData\\MASKS\\s1p05\\manual\\"
files_dir = os.listdir(path) # E:\\ResearchData\\MASKS\\s1p05-mouth\\    E:\\ResearchData\\MASKS\\manual\\s1p05\\
count = 0
for file in files_dir:
    np_img = np.asarray(PIL.Image.open(path + file).convert('RGB'))
    if np_img.shape != (1024, 1024, 3):
        print(file)
    count = count + 1