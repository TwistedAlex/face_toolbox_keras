import PIL.Image
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def generate_masks(input_dir, output_dir, limit):
    parsing_annos = [
            '0, background', '1, skin', '2, left eyebrow', '3, right eyebrow',
            '4, left eye', '5, right eye', '6, glasses', '7, left ear', '8, right ear', '9, earings',
            '10, nose', '11, mouth', '12, upper lip', '13, lower lip',
            '14, neck', '15, neck_l', '16, cloth', '17, hair', '18, hat'
        ]
    cmap = plt.get_cmap('gist_ncar', len(parsing_annos))
    new_colors = cmap(np.linspace(0, 1, len(parsing_annos)))
    new_colors[0, :] = np.array([0, 0, 0, 1.])
    new_cmap = ListedColormap(new_colors)

    path = input_dir
    files_dir = os.listdir(path) # E:\\ResearchData\\MASKS\\s1p05-mouth\\    E:\\ResearchData\\MASKS\\manual\\s1p05\\
    count = 0
    for file in files_dir:
        if count < limit:
            np_img = np.asarray(PIL.Image.open(path + file).convert('RGB'))

            r1, g1, b1 = 0, 0, 0  # Original value
            r2, g2, b2 = 255, 255, 255  # Value that we want to replace it with
            r2, g2, b2 = 255, 255, 255  # Value that we want to replace it with

            red, green, blue = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2]
            mask = (red == r1) & (green == g1) & (blue == b1)

            np_img[:, :, :3][np.invert(mask)] = [r1, g1, b1]
            np_img[:, :, :3][mask] = [r2, g2, b2]

            PIL.Image.fromarray(np_img, 'RGB').save(output_dir + file[:-4] + 'm.png')
            count = count + 1
