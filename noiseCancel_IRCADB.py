from scipy.fftpack import fft,ifft,fftshift,ifftshift
import ctypes as ct
import numpy as np
import os
import re
import math
import nibabel
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import pickle
import time
import argparse
import SimpleITK as sitk


def noise_remove_connected_region_2(sub_folder=None): # skimage.measure.label & scipy.ndimage.label
    # from skimage import morphology, measure
    from scipy.ndimage import label, generate_binary_structure
    
    # Read the segmented 3D binary volume
    for item in sub_folder:
        pred_root = item
        # connectivity=2
        for root, dirs, files in os.walk(pred_root): 
            for name in files:
                # if os.path.exists(os.path.join(root, name.replace('BinaryReconFBP','noiseCancelConnect'))):
                #     continue
                if 'BinaryReconFBP' not in name:
                    continue
                if 'optIter_4' not in name and 'optIter_9' not in name:
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
                print(np.max(opened_volume) + 1)
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
                save_path = os.path.join(root, name.replace('BinaryReconFBP','noiseCancelConnect'))
                sitk.WriteImage(img, '{}'.format(save_path))


def split_list_ordered(data_path_list, N):
    # Get the length of each part
    avg = len(data_path_list) // N
    remainder = len(data_path_list) % N
    result = []
    
    start = 0
    for i in range(N):
        # Calculate the length of the current part
        length = avg + (1 if i < remainder else 0)
        result.append(data_path_list[start:start + length])
        start += length
    
    return result

def cases_split(N_idx, N):

    data_root_list = [
        '/model/latent_diffusion/outputs/Abla1_IRCADBExp_plain_leave1out_F0/recons_singleTermOpt',
        '/model/latent_diffusion/outputs/Abla1_IRCADBExp_plain_leave1out_F1/recons_singleTermOpt',
        '/model/latent_diffusion/outputs/Abla1_IRCADBExp_plain_leave1out_F2/recons_singleTermOpt',
        '/model/latent_diffusion/outputs/Abla1_IRCADBExp_plain_leave1out_F3/recons_singleTermOpt',
        '/model/latent_diffusion/outputs/Abla1_IRCADBExp_plain_leave1out_F4/recons_singleTermOpt'
    ]
    data_path_list = []
    for data_root in data_root_list:
        temp_list = []
        for root, dirs, files in os.walk(data_root): 
            # if 'sampleEps_IRCADB_leave1out' not in root.split('/')[-3]: 
            #     continue
            # if '2nd' in root.split('/')[-3] or '3rd' in root.split('/')[-3]: 
            #     continue
            # if root.split('/')[-2] != 'samples':
            #     continue
            if 'validation_iter' not in root.split('/')[-1]:
                continue
            if root not in data_path_list:
                temp_list.append(root)
        data_path_list.extend(temp_list)

        
    data_path_list = sorted(data_path_list)

    split_data = split_list_ordered(data_path_list, N)

    # Print each part with element names
    print(f"Part {N_idx}:")
    for element in split_data[N_idx]:
        print(f" - {element}")

    noise_remove_connected_region_2(sub_folder=split_data[N_idx])

def main(args):
    cases_split(N_idx=args.N_idx, N=args.N)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='temp')
    parser.add_argument('--N', default=1, type=int)
    parser.add_argument('--N_idx', default=-1, type=int)
    args = parser.parse_args()

    main(args)