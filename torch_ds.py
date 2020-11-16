import numpy as np
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle
import argparse
import random

import SimpleITK as sitk
'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''

def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def normalizePlanes(npzarray, maxHU=600, minHU=-1200):
    maxHU = 600
    minHU = -1200
    # d3_array = np.empty([npzarray.shape[0], npzarray.shape[1], 3])
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1
    npzarray[npzarray<0] = 0
    return npzarray

class LunaDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode="train"):
        super(LunaDataset, self).__init__()
        self.df = shuffle(pd.read_csv(path))
        self.df['is_train'] = self.df['path'].apply(lambda x: int(x.startswith("train")))

        if mode == "train":
            self.df = self.df[self.df['is_train'] == 1]
        else:
            self.df = self.df[self.df['is_train'] == 0]

        print(len(self.df[self.df['class'] == 1]), len(self.df))
        ones_class = self.df[self.df['class'] == 1]
        if mode == "train":
            for a in tqdm(range(300)):
                self.df = self.df.append(ones_class)
        print(len(self.df[self.df['class'] == 1]), len(self.df))
        self.df = self.df.sample(frac=1)
        self.vW = 66
        self.start = 0
        self.end = len(self.df)

    # def __getitem__(self, index):
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # print(row)
        vW = self.vW
        ct_scan, origin, spacing = load_itk(row['path'])
        wloc = row['world_loc'][1:-1].strip().split()
        # print(wloc)
        z = int(float(wloc[0]))
        y = int(float(wloc[1]))
        x = int(float(wloc[2]))

        # plt.imshow(normalizePlanesGrey(ct_scan[z, y-vW//2: y+vW//2, x-vW//2: x+vW//2], 600), cmap=cm.gray)
        normalised = normalizePlanes(ct_scan[z-vW//2: z+vW//2, y-vW//2: y+vW//2, x-vW//2: x+vW//2]) # B 1 Z Y X
        z, y, x = normalised.shape
        normalised = np.pad(normalised, [((66-z)//2, (67-z)//2), ((66-y)//2, (67-y)//2), ((66-x)//2, (67-x)//2)])
        z_start = random.randint(0, 25)
        y_start = random.randint(0, 25)
        x_start = random.randint(0, 25)
        normalised = normalised[z_start: z_start+40, y_start: y_start+40, x_start: x_start+40]
        normalised = np.rot90(normalised, k=random.randint(0, 3), axes=(1, 2))
        if random.randint(0,1) == 0:
            normalised = np.flip(normalised, 1).copy()
        if random.randint(0,1) == 0:
            normalised = np.flip(normalised, 2).copy()

        # print(normalised.shape)
        return torch.Tensor(normalised.copy()).unsqueeze(0).detach(), int(row['class'])

    def __len__(self):
        return self.end

if __name__ == '__main__':
    KD = LunaDataset('processed_train.csv', 'test')
    it = iter(KD)
    i = 0
    for x in it:
        print(x)
        i += 1
        if i%4 == 0:
            break

