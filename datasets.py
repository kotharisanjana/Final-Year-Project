import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# %matplotlib inline
import os
import sys
import time

import SimpleITK as sitk
'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''

origins = dict()
spacings = dict()

def load_itk(filename, cache=True):
    # Reads the image using SimpleITK

    if filename in origins.keys():
        return None, origins[filename], spacings[filename]
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    origins[filename] = origin
    spacings[filename] = spacing
    return ct_scan, origin, spacing


def normalizePlanes(npzarray, maxHU=600, minHU=-1200):
  maxHU = 600
  minHU = -1200
  # d3_array = np.empty([npzarray.shape[0], npzarray.shape[1], 3])
  npzarray = (npzarray - minHU) / (maxHU - minHU)
  npzarray[npzarray>1] = 1
  npzarray[npzarray<0] = 0
  # np_r = np.copy(npzarray)
  # np_r = (np_r - -1200) / (-1000 - -1200)
  # np_r[np_r > 1] = 0
  # np_r[np_r < 0] = 0
  
  # np_g = np.copy(npzarray)
  # np_g = (np_g - -1000) / (0 - -1000)
  # np_g[np_g > 1] = 0
  # np_g[np_g < 0] = 0
  
  # np_b = np.copy(npzarray)
  # np_b = (np_b - 0) / (600 - 0)
  # np_b[np_b > 1] = 0
  # np_b[np_b < 0] = 0
  
  # d3_array[:, :, 0] = np_r
  # d3_array[:, :, 1] = np_g
  # d3_array[:, :, 2] = np_b
  # return d3_array


def locationToWorld(location, origin, spacing):
  return np.abs(location-origin)/spacing

df = pd.read_csv("candidates_V2.csv")

# print(df[df['class']==1].head())

train_files = [(f[:-4], os.path.join("train", f)) for f in os.listdir("train") if f.endswith("mhd")]
test_files = [(f[:-4], os.path.join("test", f)) for f in os.listdir("test") if f.endswith("mhd")]

train_dict = {k: v for k, v in train_files}
test_dict = {k: v for k, v in test_files}

# print(test_dict)

print(len(df))
# df = df.head(200000)

df['path'] = df['seriesuid'].apply(lambda x: train_dict.get(x, test_dict.get(x, None)))

df = df.dropna(axis=0)

s = time.time()

df['origin'] = df['path'].apply(lambda x: load_itk(x)[1])
df['spacing'] = df['path'].apply(lambda x: load_itk(x)[2])

# print(df.head())
df['world_loc'] = df.apply(lambda x: locationToWorld([x['coordZ'], x['coordY'], x['coordX']], x['origin'], x['spacing']), axis=1)

e = time.time() - s
print(df.head(), e)

df.to_csv('processed_train.csv')

