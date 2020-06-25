from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from glob import glob
from pathlib import Path
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2
import numpy as np
import shutil
from torchvision import transforms

IMAGE_SIZE = (256, 256)
SOURCE_DIR = os.path.join("data",
                          "Lung_Segmentation")
MASK_DIR = os.path.join(SOURCE_DIR,
                        "ManualMask")
LEFT_MASK_DIR = os.path.join(MASK_DIR,
                             "leftMask")
RIGHT_MASK_DIR = os.path.join(MASK_DIR,
                              "rightMask")


def findMasksPath(left_mask, right_mask):
    left_pngs = glob(os.path.join(left_mask, "*.png"))
    right_pngs = glob(os.path.join(right_mask, "*.png"))
    return left_pngs, right_pngs


def combineMasks(left_mask, right_mask):
    left = cv2.imread(left_mask, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_mask, cv2.IMREAD_GRAYSCALE)
    left = cv2.resize(left, IMAGE_SIZE)
    right = cv2.resize(right, IMAGE_SIZE)
    combined_mask = np.maximum(left, right)
    combined_mask_dilate = cv2.dilate(combined_mask, np.ones((15, 15), dtype=np.uint8), iterations=1)
    return combined_mask, combined_mask_dilate


def saveMasks(p):
    if not os.path.exists(p):
        os.mkdir(p)

    left, right = findMasksPath(LEFT_MASK_DIR, RIGHT_MASK_DIR)

    for l, r in zip(left, right):
        name = l.split('/')[-1]
        mask, dilate_mask = combineMasks(l, r)
        cv2.imwrite(os.path.join(p, name), mask)

    shutil.rmtree(MASK_DIR)
    return


def lungDataProcess(export_images=False):
    if not os.path.exists(os.path.join(SOURCE_DIR, 'masks')):
        saveMasks(os.path.join(SOURCE_DIR, 'masks'))
    base_path = Path('data') / 'Lung_Segmentation'
    all_images_df = pd.DataFrame({'path': list(base_path.glob('**/*.*p*g'))})
    all_images_df['modality'] = all_images_df['path'].map(lambda x: x.parent.stem)
    all_images_df['source'] = all_images_df['path'].map(lambda x: x.stem.split('_')[0])
    all_images_df['image_id'] = all_images_df['path'].map(lambda x: '_'.join(x.stem.split('_')[1:2]))
    all_images_df = all_images_df[all_images_df['modality'].isin(['masks', 'CXR_png'])]
    print(all_images_df['modality'].value_counts())

    flat_images_df = all_images_df.pivot_table(index=['source', 'image_id'],
                              columns='modality',
                              values='path',
                              aggfunc='first').reset_index().sort_values('image_id').dropna()

    if export_images:
        fig, (a_axs, b_axs) = plt.subplots(2, 4, figsize=(20, 5))
        for a_ax, b_ax, (_, c_row) in  zip(a_axs, b_axs, flat_images_df.sample(20).iterrows()):
            a_img = imread(c_row['CXR_png'])
            a_ax.imshow(a_img)
            a_ax.set_title(c_row['source'])
            b_img = imread(c_row['masks'])
            b_ax.imshow(b_img)

        fig, (a_axs, b_axs) = plt.subplots(2, 4, figsize=(20, 5))
        for a_ax, b_ax, (_, c_row) in zip(a_axs, b_axs, flat_images_df.sample(20).iterrows()):
            a_img = imread(c_row['CXR_png'])
            a_ax.hist(a_img.ravel())
            a_ax.set_title(c_row['source'])
            b_img = imread(c_row['masks'])
            b_ax.hist(b_img.ravel())

    img = flat_images_df.CXR_png.map(str).copy()
    seg = flat_images_df.masks.map(str).copy()
    return img.to_frame('image_path'), seg.to_frame('image_path')


def prostateDataProcess(export_images=False):
    base_path = os.path.join('data', 'Prostate_Segmentation')
    images, segmentations = np.load(os.path.join(base_path, "X_train.npy")), np.load(os.path.join(base_path, "y_train.npy"))
    images = images.transpose((0, 3, 1, 2))
    images = images.reshape((images.shape[0], images.shape[2], images.shape[3]))
    segmentations = segmentations.transpose((0, 3, 1, 2))
    segmentations = segmentations.reshape((segmentations.shape[0], segmentations.shape[2], segmentations.shape[3]))

    indx = []
    for i, (im, seg) in enumerate(zip(images, segmentations)):
        if len(np.unique(seg)) < 2:
            indx.append(i)
    print("removed {:} empty segmentation maps".format(len(indx)))
    images = np.delete(images, indx, axis=0)
    segmentations = np.delete(segmentations, indx, axis=0)

    if export_images:
        im = [images[200, :], images[1200, :]]
        seg = [segmentations[200, :], segmentations[1200, :]]
        fig, (a_axs, b_axs) = plt.subplots(2, 2, figsize=(6, 6))
        for (a_ax, b_ax, x, y) in zip(a_axs, b_axs, im, seg):
            a_ax.imshow(x.reshape(x.shape[1], x.shape[2]))
            a_ax.set_title("Orig")
            a_ax.axis('off')
            b_ax.imshow(y.reshape(y.shape[1], y.shape[2]))
            b_ax.set_title("Seg")
            b_ax.axis('off')
        fig.tight_layout()

        fig, (a_axs, b_axs) = plt.subplots(2, 2, figsize=(20, 5))
        for (a_ax, b_ax, x, y) in zip(a_axs, b_axs, im, seg):
            a_ax.hist(x.ravel())
            a_ax.set_title("Orig")
            b_ax.hist(y.ravel())
        fig.tight_layout(pad=0.5)
    return images, segmentations


class LungSeg(Dataset):
    def __init__(self, img_df, seg_df, transforms=None):
        self.imgs = img_df["image_path"].to_list()
        self.seg = seg_df["image_path"].to_list()
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.imgs[index])
        image = image.convert('L')
        mask = Image.open(self.seg[index])
        mask = mask.convert('L')
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        return (image, mask)

    def __len__(self):
        return len(self.imgs)


class ProstateSeg(Dataset):
    def __init__(self, imgs, segs, transforms=None):
        self.imgs = imgs
        self.segs = segs
        self.transforms = transforms

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        image = (self.imgs[index, :] * 255).astype('uint8')
        mask = (self.segs[index, :] * 255).astype('uint8')
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        return (image, mask)