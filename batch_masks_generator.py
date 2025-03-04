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

    if idx < 0:
        # masks=out[0]==0
        # np_img = np.asarray(PIL.Image.open(input_dir + image_name).convert('RGB'))
        plt.imsave(output_dir + image_name, out[0], cmap=new_cmap)
    else:
        old_out = out[0]
        old_out = np.where(old_out == 18, 0, old_out)
        if idx == 0:  # ear
            old_out = np.where(old_out == 7, 18, old_out)
            old_out = np.where(old_out == 8, 18, old_out)
            old_out = np.where(old_out == 9, 18, old_out)
        elif idx == 1:  # mouth
            old_out = np.where(old_out == 11, 18, old_out)
            old_out = np.where(old_out == 12, 18, old_out)
            old_out = np.where(old_out == 13, 18, old_out)
        elif idx == 2:  # nose
            old_out = np.where(old_out == 10, 18, old_out)
        elif idx == 3:  # eye
            old_out = np.where(old_out == 4, 18, old_out)
            old_out = np.where(old_out == 5, 18, old_out)
            old_out = np.where(old_out == 6, 18, old_out)
        elif idx == 4:  # eyebrow
            old_out = np.where(old_out == 2, 18, old_out)
            old_out = np.where(old_out == 3, 18, old_out)
        elif idx == 5:  # neck
            old_out = np.where(old_out == 14, 18, old_out)
            old_out = np.where(old_out == 15, 18, old_out)

        old_out = np.where(old_out != 18, 0, old_out)

        plt.imsave(output_dir + f'{output_name}m.png', old_out, cmap=new_cmap)

        # vis_im = cv2.addWeighted(im, 0.1, cv2.cvtColor(old_out, cv2.COLOR_RGB2BGR), 0.9, 20)
        # plt.imshow(vis_im)


def main(args):
    # input_dir = "E:\\ResearchData\\stylegan-psi05\\training\\Pos\\"
    # output_dir = "E:\\ResearchData\\MASKS\\s1p05\\"
    prefix = ""
    modes = ["ear", "mouth", "nose", "eye", "eyebrow", "neck"]
    # modes = ["segmentation"]
    selected_modes = [0, ]
    all_images = os.listdir(args.input_dir)
    print(args.input_dir)
    # Test images are obtained on https://www.pexels.cosm/
    count = 0
    limit = args.mode
    limit_low = -1
    limit_high = args.high
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
        if args.mode > 0:
            for idx in selected_modes:
                if limit_low < count < limit_high:
                    save_mask_for(
                        prs,
                        new_cmap,
                        image_name,
                        args.input_dir,
                        args.output_dir + modes[idx] + "//",
                        idx
                    )
        else:
            save_mask_for(
                prs,
                new_cmap,
                image_name,
                args.input_dir,
                args.output_dir,
                -1
            )
        count += 1


# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch GAIN Training')
parser.add_argument('--input_dir', help='path to the input idr', type=str)
parser.add_argument('--output_dir', help='path to the outputdir', type=str)
parser.add_argument('--mode', help='prefix str to add to the output file name', type=int)
parser.add_argument('--high', help='prefix str to add to the output file name', type=int)
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
