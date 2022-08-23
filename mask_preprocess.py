import PIL.Image
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

path = "E:\\ResearchData\\MASKS\\manual\\s1p05\\accessories\\"
files_dir = os.listdir(path)  # E:\\ResearchData\\MASKS\\s1p05-mouth\\    E:\\ResearchData\\MASKS\\manual\\s1p05\\
count = 0
limit = 18
for file in files_dir:
    if count < limit:
        np_img = np.asarray(PIL.Image.open(path + file).convert('RGB'))
        r1, g1, b1 = 0, 0, 0  # Original value
        r2, g2, b2 = 1, 1, 1  # Value that we want to replace it with

        red, green, blue = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        try:
            np_img[:, :, :3][mask] = [r2, g2, b2]
        except IndexError:
            print("index" + file)
            pass
        except OSError:
            print("OSError" + file)
        plt.imsave(path + file, np_img)
        count += 1
