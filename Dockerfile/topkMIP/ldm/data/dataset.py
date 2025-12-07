import os
import torch
import nibabel
import re
# import monai
import numpy as np
# from matplotlib import pyplot as plt
import random
import SimpleITK as sitk
import math
import cv2

def save_nifti(img, img_path):
    pair_img = nibabel.Nifti1Pair(img,np.eye(4))
    nibabel.save(pair_img,img_path)



class Dataset3D_diffusion_infer(torch.utils.data.Dataset): # cropped sub volume as output
    def __init__(self, root_path, clip_min, clip_max, name=None):
        'Initialization'
        self.name = name
        self.clip_min = clip_min
        self.clip_max = clip_max
        test_patch_path = {'projCTMIP':[]}

        for root, dirs, files in os.walk(root_path): 
            for name in files:
                test_patch_path['projCTMIP'].append(os.path.join(root,name))

        test_patch_path['projCTMIP'] = sorted(test_patch_path['projCTMIP'])

        self.list_IDs = test_patch_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['projCTMIP'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        proj_ctmip = nibabel.load(self.list_IDs['projCTMIP'][index]).get_fdata().transpose(2,0,1) # (6*scales,256,256)
        path = self.list_IDs['projCTMIP'][index].split('/')[-1]

        proj_ctmip = np.clip(proj_ctmip, self.clip_min, self.clip_max)
        proj_ctmip = (proj_ctmip-self.clip_min) / (self.clip_max-self.clip_min+1e-7)

        proj_ctmip = torch.from_numpy(proj_ctmip)[None].type(torch.float32) # (1,6*scales,256,256)

        'batch'
        # proj_ctmip = proj_ctmip.permute(1,0,2,3)
        proj_ctmip = proj_ctmip.reshape(-1,1,256,256)

        return tuple((proj_ctmip, None, path))
