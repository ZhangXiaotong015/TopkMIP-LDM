# from scipy.fftpack import fft,ifft,fftshift,ifftshift
# import ctypes as ct
import numpy as np
import os
# import re
# import math
# import nibabel
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import scipy.sparse as sp
# import pickle
# import time
import argparse
import SimpleITK as sitk


def noise_remove_connected_region_2(sub_folder=None, save_root=None, iter_idx=None): # skimage.measure.label & scipy.ndimage.label
    # from skimage import morphology, measure
    from scipy.ndimage import label, generate_binary_structure
    
    # Read the segmented 3D binary volume
    for root, dirs, files in os.walk(sub_folder): 
        for name in files:
            if f'optIter_{iter_idx}' not in name:
                continue

            segmented_volume = sitk.ReadImage(os.path.join(root, name))
            origin = segmented_volume.GetOrigin()
            space = segmented_volume.GetSpacing()
            direction = segmented_volume.GetDirection()
            volume_unit = np.round(space[0]*space[1]*space[2], 4) # mm3

            segmented_volume = sitk.GetArrayFromImage(segmented_volume)
            structure = generate_binary_structure(3, 1)
            opened_volume, num_features = label(segmented_volume, structure=structure)

            component_size_list = []
            accumulated_img = np.zeros_like(segmented_volume)
            # print(np.max(opened_volume) + 1)
            for label_item in range(1, np.max(opened_volume) + 1):
                component_mask = np.uint8(opened_volume == label_item)
                component_size = np.sum(component_mask)
                accumulated_img[opened_volume == label_item] = component_size
                component_size_list.append(component_size)

            accumulated_img = (accumulated_img-accumulated_img.min()) / (accumulated_img.max()-accumulated_img.min())
            accumulated_img[accumulated_img>0.01] = 1
            accumulated_img[accumulated_img<=0.01] = 0

            img = sitk.GetImageFromArray(accumulated_img)
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(space)
            os.makedirs(save_root, exist_ok=True)
            save_path = os.path.join(save_root, name.replace('BinaryReconFBP','noiseCancelConnect'))
            sitk.WriteImage(img, '{}'.format(save_path))


def pipeline(data_root, save_root, iter_idx):

    noise_remove_connected_region_2(sub_folder=data_root, save_root=save_root, iter_idx=iter_idx)

def noiseCancel(data_root, save_root, iter_idx):
    pipeline(data_root, save_root, iter_idx)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='temp')
    parser.add_argument('--N', default=1, type=int)
    parser.add_argument('--N_idx', default=-1, type=int)
    args = parser.parse_args()

    noiseCancel(args)