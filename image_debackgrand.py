import warnings
import PIL.Image
warnings.filterwarnings("ignore")
import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os


def save_mask_for(image_name, segmentation_dir, orig_path, output_dir, idx):

    out = cv2.imread(segmentation_dir + image_name)
    r1, g1, b1 = 0, 0, 0  # Original value
    r2, g2, b2 = 0, 0, 0
    if idx < 0:
        # get masks from the segmentation where value == black(background)
        red, green, blue = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        mask = (red == r1) & (green == g1) & (blue == b1)

        np_img = np.asarray(cv2.imread(orig_path + image_name))
        np_img[:, :, :3][mask] = [r2, g2, b2]
        cv2.imwrite(output_dir + image_name, np_img)


def main(args):
    segmentation_dir = "E:\\ResearchData\\MASKS\\s1p05\\segmentation\\"
    orig_path = "E:\\ResearchData\\stylegan-psi05\\training\\Pos\\"
    output_dir = "E:\\ResearchData\\MASKS\\s1p05\\debackground\\"

    all_images = os.listdir(args.segmentation_dir)

    count = 0
    all_images.sort()
    for image_name in all_images:
        if count < 1000000:
            save_mask_for(
                image_name,
                args.segmentation_dir,
                args.orig_path,
                args.output_dir,
                -1
            )
        count += 1


# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch GAIN Training')
parser.add_argument('--segmentation_dir', help='path to the input idr', type=str)
parser.add_argument('--orig_path', help='path to the outputdir', type=str)
parser.add_argument('--output_dir', help='prefix str to add to the output file name', type=str)
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
