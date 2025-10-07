"""
TransUNet
└── data
    └──ToothSegmDataset
        └── trainset_valset (base_dir)
            ├── 0
            ├── 1
            │   ├── train
            │   │   ├── 121_11
            │   │        ├── 000_mask.jpg
            │   │        ├── 000_rgb.jpg
            │   │        └── ...
            │   └── val
            │       ├── 220_11
            │       │    ├── 000_mask.jpg
            │       │    ├── 000_rgb.jpg
            │       │    └── ...
            │       ├── ...
            │       └── 25839_11
            └── ...       ├── 000_mask.jpg
                          ├── 000_rgb.jpg
                          └── ...
How to read its mask?
import cv2
import numpy as np
def read_mask(mask_path):
   # value:1 upper
   # value:2 lower
   # value:3 a tooth
   mask_img = cv2.imread(mask_path,-1)
   mask_upper = np.zeros_like(mask_img)
   mask_upper = np.where(mask_img==1,255,0)
   mask_lower = np.zeros_like(mask_img)
   mask_lower = np.where(mask_img==2,255,0)
   mask_tooth = np.zeros_like(mask_img)
   mask_tooth = np.where(mask_img==3,255,0)
   return mask_upper,mask_lower,mask_tooth
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from scipy.ndimage import zoom

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2


# -------------------------------
# Data Augmentation Functions
# -------------------------------
def random_rot_flip(image, label):
    """Random rotation (90 degrees) and flip"""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    """Random rotation with arbitrary angle"""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=1, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)  # nearest for mask
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Apply augmentation
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # Resize if needed
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  # bilinear for RGB
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # nearest for mask

        # Convert to torch tensors
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # (C,H,W)
        label = torch.from_numpy(label.astype(np.float32))

        # CHANGE: if label has any values greater than 3 like 4 and 5, convert them to 0
        label = torch.where(label > 3, torch.tensor(0.0), label)

        sample = {'image': image, 'label': label.long()}
        return sample


# -------------------------------
# Mask Reader
# -------------------------------
def read_mask(mask_path):
    """
    Read mask and extract categories.
    Values:
        1 -> upper
        2 -> lower
        3 -> tooth
    """
    mask_img = cv2.imread(mask_path, -1)

    # value:1 upper
    # value:2 lower
    # value:3 a tooth
    # mask_upper = np.zeros_like(mask_img)
    # mask_upper = np.where(mask_img == 1, 255, 0)
    #
    # mask_lower = np.zeros_like(mask_img)
    # mask_lower = np.where(mask_img == 2, 255, 0)
    #
    # mask_tooth = np.zeros_like(mask_img)
    # mask_tooth = np.where(mask_img == 3, 255, 0)

    # convert directly into class label map (0 background, 1 upper, 2 lower, 3 tooth)
    # mask_img is already in that form if dataset annotation is correct
    return mask_img


# -------------------------------
# ToothSegmentation Dataset
# -------------------------------
class ToothSegmDataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None):
        """
        base_dir: path to 'ToothSegmDataset/trainset_valset'
        """
        self.base_dir = base_dir
        self.split = split
        self.transform = transform

        # collect all samples (folders inside train/val)
        self.sample_list = []
        for tooth_id in os.listdir(base_dir):
            tooth_dir = os.path.join(base_dir, tooth_id)  # e.g., ToothSegmDataset/trainset_valset/0
            if not os.path.isdir(tooth_dir):
                continue

            split_dir = os.path.join(tooth_dir, split)  # e.g., ToothSegmDataset/trainset_valset/0/train
            for case_name in os.listdir(split_dir):
                case_dir = os.path.join(split_dir, case_name)  # e.g., ToothSegmDataset/trainset_valset/0/train/121_11
                if not os.path.isdir(case_dir):
                    continue

                rgb_files = sorted([f for f in os.listdir(case_dir) if f.endswith("_rgb.jpg")])
                for rgb_file in rgb_files:
                    mask_file = rgb_file.replace("_rgb.jpg", "_mask.jpg")
                    self.sample_list.append({
                        "image": os.path.join(case_dir, rgb_file),
                        "label": os.path.join(case_dir, mask_file),
                        "tooth_id": tooth_id,
                        "case_name": case_name
                    })

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_info = self.sample_list[idx]

        # load image and mask
        image = cv2.imread(sample_info["image"])  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = read_mask(sample_info["label"])

        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        sample["case_name"] = sample_info["case_name"]
        sample["tooth_id"] = int(sample_info["tooth_id"])

        # get the min and max number of label figure
        # label_figure = sample['label'].numpy()
        # min_value_in_label_figure = np.min(label_figure)
        # max_value_in_label_figure = np.max(label_figure)
        # TODO: Ask why this is 0,4?
        # print("min and max value in label figure: ", min_value_in_label_figure, max_value_in_label_figure) # 0,4

        return sample  # image: (3, 224, 224), label: (224, 224)
