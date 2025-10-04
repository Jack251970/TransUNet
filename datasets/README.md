# Data Preparing

1. Access to the synapse multi-organ dataset:
   1. Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/) and download the dataset. Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases.
   2.  You can also send an Email directly to jienengchen01 AT gmail.com to request the preprocessed data for reproduction.
2. The directory structure of the whole project is as follows:

```bash
.
├── TransUNet
│   ├──datasets
│   │       └── dataset_*.py
│   ├──train.py
│   ├──test.py
│   └──...
├── model
│   └── vit_checkpoint
│       └── imagenet21k
│           ├── R50+ViT-B_16.npz
│           └── *.npz
└── data
    └──Synapse
        ├── test_vol_h5
        │   ├── case0001.npy.h5
        │   └── *.npy.h5
        └── train_npz
            ├── case0005_slice000.npz
            └── *.npz
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
```
