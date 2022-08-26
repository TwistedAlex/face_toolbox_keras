import cv2
import PIL.Image
import os
import numpy as np
from matplotlib import pyplot as plt
import pathlib
from masks_from_manual import generate_masks
import random


def pick_masks(input_dir, output_dir, limit, num=None):
    files_dir = os.listdir(input_dir)  # E:\\ResearchData\\MASKS\\s1p05-mouth\\    E:\\ResearchData\\MASKS\\manual\\s1p05\\
    count = 0
    for file in files_dir:
        if count < limit:
            np_img = cv2.cvtColor(cv2.imread(input_dir + file), cv2.COLOR_BGR2GRAY)
            numLabels, labels, stats,centroids = cv2.connectedComponentsWithStats(np_img, connectivity=8)
            # print(numLabels)
            #
            # print(np.unique(labels))
            # # print(labels.shape)
            # print(stats.shape)
            # print(centroids)
            if num is None or numLabels - 1 <= num:
                pass
            else:
                random_picked = random.sample(range(numLabels - 1), num)
                print(numLabels)
                print(random_picked)
                for idx in range(numLabels):
                    if idx in random_picked:
                        labels = np.where(labels == (idx + 1), 1, labels)
                    else:
                        labels = np.where(labels == (idx + 1), 0, labels)
            cv2.imwrite(output_dir + file, np.array(labels * 255, dtype=np.uint8))
            count += 1


def merge_masks(inputs, output_dir, limit):
    files_dir = os.listdir(inputs[0])
    count = 0
    for file in files_dir:
        if count < limit:
            output_np = np.full((1024, 1024), 0, dtype=np.uint8)
            for idx in range(len(inputs)):
                im = cv2.imread(inputs[idx] + file)
                mask_pos = im[:,:,0] == 255
                output_np[mask_pos] = 255
            cv2.imwrite(output_dir + file, output_np)
            count += 1


def resize_masks(input_dir, output, scale_ratio, limit):
    files_dir = os.listdir(input_dir)
    count = 0
    for file in files_dir:
        if count < limit:
            output_np = np.full((1024, 1024), 0, dtype=np.uint8)

            np_img = cv2.cvtColor(cv2.imread(input_dir + file), cv2.COLOR_BGR2GRAY)
            numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(np_img, connectivity=8)
            for idx in range(len(stats)):
                if idx == 0:
                    continue
                start_p = (int(stats[idx][0]), int(stats[idx][1]))
                end_p = (int(stats[idx][0] + stats[idx][2] * scale_ratio),
                         int(stats[idx][1] + stats[idx][3] * scale_ratio))
                output_np = cv2.rectangle(output_np, start_p, end_p, 255, -1)
            cv2.imwrite(output + file, output_np)
            count += 1


def classify_masks(input_dir, output_dir, output_wrinkles_dir, target_modes, class_masks_dir, limit):
    files_dir = os.listdir(input_dir)
    len_modes = len(target_modes)
    count = 0

    for file in files_dir:
        if count < limit:
            output_np = np.full((len_modes + 1, 1024, 1024), 0, dtype=np.uint8)
            np_img = cv2.cvtColor(cv2.imread(input_dir + file), cv2.COLOR_BGR2GRAY)
            numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(np_img, connectivity=8)

            for idx in range(len(stats)):
                belong_to_specific_region = False
                if idx == 0:
                    continue
                cur_region = np.full((1024, 1024), 0, dtype=np.uint8)
                start_p = (int(stats[idx][0]), int(stats[idx][1]))
                end_p = (int(stats[idx][0] + stats[idx][2]),
                         int(stats[idx][1] + stats[idx][3]))
                cur_np = cv2.rectangle(cur_region, start_p, end_p, 255, -1)
                mask = cur_np == 255

                for mode_idx in range(len_modes):
                    print(class_masks_dir + target_modes[mode_idx] + "\\"+file)
                    mode_np = cv2.imread(class_masks_dir + target_modes[mode_idx] + "\\"+file)
                    intersection = np.logical_and(cur_np, mode_np[:,:,0])
                    if intersection.any():
                        # merge cur_np into the desired np, init desired np all black(0),
                        output_np[mode_idx][mask] = 255
                        belong_to_specific_region = True
                if not belong_to_specific_region:
                    output_np[len_modes][mask] = 255

            count += 1
            for mode_idx in range(len_modes):
                pathlib.Path(output_dir + target_modes[mode_idx] + "\\").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(output_dir + target_modes[mode_idx] + "\\" + file, output_np[mode_idx])
            cv2.imwrite(output_dir + "wrinkles_others\\" + file, output_np[len_modes])


def remove_bg(input_dir, output_wrinkles_dir, target_modes, class_masks_dir, limit):
    files_dir = os.listdir(input_dir)
    len_modes = len(target_modes)
    count = 0

    for file in files_dir:
        if count < limit:
            output_np = np.full((len_modes + 1, 1024, 1024), 0, dtype=np.uint8)
            np_img = cv2.cvtColor(cv2.imread(input_dir + file), cv2.COLOR_BGR2GRAY)
            numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(np_img, connectivity=8)

            for idx in range(len(stats)):
                belong_to_specific_region = False
                if idx == 0:
                    continue
                cur_region = np.full((1024, 1024), 0, dtype=np.uint8)
                start_p = (int(stats[idx][0]), int(stats[idx][1]))
                end_p = (int(stats[idx][0] + stats[idx][2]),
                         int(stats[idx][1] + stats[idx][3]))
                cur_np = cv2.rectangle(cur_region, start_p, end_p, 255, -1)
                mask = cur_np == 255

                for mode_idx in range(len_modes):
                    print(class_masks_dir + target_modes[mode_idx] + "\\"+file)
                    mode_np = cv2.imread(class_masks_dir + target_modes[mode_idx] + "\\"+file)
                    intersection = np.logical_and(cur_np, mode_np[:,:,0])
                    if intersection.any():
                        # merge cur_np into the desired np, init desired np all black(0),
                        output_np[mode_idx][mask] = 255
                        belong_to_specific_region = True
                if not belong_to_specific_region:
                    output_np[len_modes][mask] = 255

            count += 1
            for mode_idx in range(len_modes):
                pathlib.Path(output_wrinkles_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(output_wrinkles_dir + "\\" + file, output_np[mode_idx])


def main():
    modes = ["waterdrops\\", "wrinkles_others_orig\\", "hair\\", ]
    draft_dir = "E:\\ResearchData\\MASKS\\manual\\s1p05\\"
    input_dir = "E:\\ResearchData\\MASKS\\manual\\s1p05\\input\\"
    output_dir = "E:\\ResearchData\\MASKS\\manual\\s1p05\\output\\"
    merge_dir = output_dir + "merged\\"
    resize_merge_dir = output_dir + "resizedmerged\\"
    pathlib.Path(merge_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(resize_merge_dir).mkdir(parents=True, exist_ok=True)
    limit = 500
    max_patches = 100

    for idx in range(len(modes)):
        print(modes[idx])
        pathlib.Path(output_dir + modes[idx]).mkdir(parents=True, exist_ok=True)
        pathlib.Path(input_dir + modes[idx]).mkdir(parents=True, exist_ok=True)
        generate_masks(draft_dir + modes[idx], input_dir + modes[idx], limit)

    wrinkles_dir = "E:\\ResearchData\\MASKS\\manual\\s1p05\\input\\wrinkles_others_orig\\"
    output_wrinkles_dir = "E:\\ResearchData\\MASKS\\manual\\s1p05\\input\\wrinkles_others\\"
    target_modes = ["ear", "mouth", "nose", "eye", "eyebrow", "neck"]
    class_masks_dir = "E:\\ResearchData\\MASKS\\s1p05\\"

    classify_masks(wrinkles_dir, input_dir, output_wrinkles_dir, target_modes, class_masks_dir, limit)

    all_modes = ["ear", "mouth", "nose", "eye", "eyebrow", "neck", "waterdrops", "wrinkles_others", "hair"]

    for idx in range(len(all_modes)):
        pick_masks(input_dir + all_modes[idx] + "\\", output_dir + all_modes[idx] + "\\", limit, num=max_patches)
    merge_masks([output_dir + mo + "\\" for mo in all_modes], merge_dir, limit)
    # resize_masks(merge_dir, resize_merge_dir, 0.5, limit)

    merged_dir = "E:\\ResearchData\\MASKS\\manual\\s1p05\\output\\merged\\"
    output_dir="E:\\ResearchData\\MASKS\\manual\\s1p05\\output\\"
    output_merged_dir="E:\\ResearchData\\MASKS\\manual\\s1p05\\output\\debg_merged\\"

    target_modes=["background"]
    limit=500
    remove_bg(merged_dir, output_merged_dir, target_modes, class_masks_dir, limit)

if __name__ == '__main__':

    main()
