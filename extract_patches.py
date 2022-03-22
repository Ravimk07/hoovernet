"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
import tqdm
import pathlib

import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from dataset import get_dataset

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = True
    #
    # win_size = [540, 540]
    # step_size = [164, 164]
    win_size = [540, 540]
    step_size = [350, 350]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17 or consep.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "consep"
    # save_root = "dataset/training_data/%s/" % dataset_namere
    save_root = "../Research/input_image/consep"
    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "train": {
            "img": (".jpg", "../Research/input_image/train/"),
            "ann": (".mat", "../Research/input_image/train/mat/"),
        },
        "valid": {
            "img": (".jpg", "../Research/input_image/val/"),
            "ann": (".mat", "../Research/input_image/val/mat"),
        }
    }

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        out_dir = "%s/%s/%dx%d_%dx%d/" % (
            save_root,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )
        # data = ["14_14","14_16","13_14","13_15"]
        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem
            temp = base_name.split("-")[1]
            # if temp not in data:
            #     continue
            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            )
            # img = np.resize(np.asarray(img),(1024,1024,3))
            # ann = np.resize(np.asarray(ann), (1024, 1024, 2))

            # *
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            # *
            pbarx.update()
        pbarx.close()
