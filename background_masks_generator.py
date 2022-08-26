import warnings
import PIL.Image
warnings.filterwarnings("ignore")
import cv2
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from models.parser import face_parser
import argparse
import os


def save_mask_for(prs, new_cmap, image_name, input_dir, output_dir, idx):
    print(image_name)
    im = cv2.imread(input_dir + image_name)[..., ::-1]
    output_name = image_name[:-4]

    out = prs.parse_face(im)

    old_out = out[0]
    old_out = np.where(old_out == 18, 1, old_out)
    old_out = np.where(old_out == 0, 18, old_out)
    old_out = np.where(old_out != 0, 0, old_out)

    plt.imsave(output_dir + f'{output_name}m.png', old_out, cmap=new_cmap)

        # vis_im = cv2.addWeighted(im, 0.1, cv2.cvtColor(old_out, cv2.COLOR_RGB2BGR), 0.9, 20)
        # plt.imshow(vis_im)


def main(args):
    all_images = os.listdir(args.input_dir)
    print(args.input_dir)
    # Test images are obtained on https://www.pexels.cosm/
    count = 0
    limit = True
    limit_low = -1
    limit_high = 1500
    all_images.sort()
    prs = face_parser.FaceParser()
    parsing_annos = [
        '0, background', '1, skin', '2, left eyebrow', '3, right eyebrow',
        '4, left eye', '5, right eye', '6, glasses', '7, left ear', '8, right ear', '9, earings',
        '10, nose', '11, mouth', '12, upper lip', '13, lower lip',
        '14, neck', '15, neck_l', '16, cloth', '17, hair', '18, hat'
    ]

    # get discrete colormap
    cmap = plt.get_cmap('gist_ncar', len(parsing_annos))
    new_colors = cmap(np.linspace(0, 1, len(parsing_annos)))
    new_colors[0, :] = np.array([0, 0, 0, 1.])
    new_cmap = ListedColormap(new_colors)

    for image_name in all_images:
        if limit_low < count < limit_high:
            save_mask_for(
                prs,
                new_cmap,
                image_name,
                args.input_dir,
                args.output_dir ,
                0
            )
        count += 1


# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch GAIN Training')
parser.add_argument('--input_dir', help='path to the input idr', type=str)
parser.add_argument('--output_dir', help='path to the outputdir', type=str)
parser.add_argument('--prefix', help='prefix str to add to the output file name', type=str)
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
