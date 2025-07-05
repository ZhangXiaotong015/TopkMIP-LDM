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


class Dataset3D_EnDe_train(torch.utils.data.Dataset): # auto encoder training
    def __init__(self, root_path, folder_projVessel, fold_num, fold_idx, clip_min, clip_max, fixed_cases_in_trainset, name=None):
        'Initialization'
        self.name = name
        self.clip_min = 0
        self.clip_max = 400
        train_patch_path = {'projVessel':[]}
        train_folder = []

        ## Leave-one-out cross validation with 15 fixed cases in training set. 
        case_idx = np.arange(1,21)
        if len(fixed_cases_in_trainset)>0:
            new_list = [x for x in list(case_idx) if x not in fixed_cases_in_trainset]
            case_idx = np.array(new_list)
        random.seed(0)
        random.shuffle(case_idx)
        case_idx_split = np.split(np.array(case_idx), fold_num)
        test_folder = list(case_idx_split[fold_idx])

        for root, dirs, files in os.walk(root_path): 
            if root.split('/')[-1]==folder_projVessel: 
                for name in files:
                    if int(re.findall(r"\d+",name)[0]) not in test_folder:
                        train_patch_path['projVessel'].append(os.path.join(root,name))
                        if int(re.findall(r"\d+",name)[0]) not in train_folder:
                            train_folder.append(int(re.findall(r"\d+",name)[0]))

        self.list_IDs = train_patch_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['projVessel'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        proj_Vessel = nibabel.load(self.list_IDs['projVessel'][index]).get_fdata().transpose(2,0,1) # (V*scales,256,256)
        path = self.list_IDs['projVessel'][index].split('/')[-1]

        proj_Vessel = np.clip(proj_Vessel, 0, 256)
        proj_Vessel = (proj_Vessel-0) / (256-0+1e-7)

        proj_Vessel = torch.from_numpy(proj_Vessel).type(torch.float32)
        proj_Vessel = proj_Vessel.reshape(-1,3,256,256)

        return tuple((proj_Vessel, path))


class Dataset3D_diffusion_train(torch.utils.data.Dataset): # diffusion u-net training
    def __init__(self, root_path, folder_projCTtopkMIP, folder_projVessel, fold_num, fold_idx, clip_min, clip_max, fixed_cases_in_trainset, name=None):
        'Initialization'
        self.name = name
        self.clip_min = 0
        self.clip_max = 400
        self.folder_projCTtopkMIP = folder_projCTtopkMIP
        self.folder_projVessel = folder_projVessel
        train_patch_path = {'mip':[], 'projVessel':[]}
        train_folder = []

        case_idx = np.arange(1,21)
        if len(fixed_cases_in_trainset)>0:
            new_list = [x for x in list(case_idx) if x not in fixed_cases_in_trainset]
            case_idx = np.array(new_list)
        random.seed(0)
        random.shuffle(case_idx)
        case_idx_split = np.split(np.array(case_idx), fold_num)
        test_folder = list(case_idx_split[fold_idx])

        for root, dirs, files in os.walk(root_path): 
            if root.split('/')[-2]!='3DircadbNifti_newLiverMask':
                continue
            if root.split('/')[-1]==folder_projCTtopkMIP: 
                for name in files:
                    if int(re.findall(r"\d+",name)[0]) not in test_folder:
                        train_patch_path['mip'].append(os.path.join(root,name))
                        if int(re.findall(r"\d+",name)[0]) not in train_folder:
                            train_folder.append(int(re.findall(r"\d+",name)[0]))

            elif root.split('/')[-1]==folder_projVessel: 
                for name in files:
                    if int(re.findall(r"\d+",name)[0]) not in test_folder:
                        train_patch_path['projVessel'].append(os.path.join(root,name))

        train_patch_path['mip'] = sorted(train_patch_path['mip'])
        train_patch_path['projVessel'] = sorted(train_patch_path['projVessel'])

        self.list_IDs = train_patch_path
        # print(train_patch_path['mip'])
        # print(train_patch_path['projVessel'])

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['projVessel'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        proj_ctmip = nibabel.load(self.list_IDs['mip'][index]).get_fdata().transpose(2, 0, 1) # (6*scales,256,256)
        # proj_vessel = nibabel.load(self.list_IDs['projVessel'][index]).get_fdata().transpose(2,0,1) # (6,256,256)
        proj_vessel = nibabel.load(self.list_IDs['mip'][index].replace(self.folder_projCTtopkMIP, self.folder_projVessel)).get_fdata().transpose(2,0,1) # (6,256,256)
        path = self.list_IDs['mip'][index].split('/')[-1]

        print('index:', index)
        print(self.list_IDs['mip'][index])
        print(self.list_IDs['mip'][index].replace(self.folder_projCTtopkMIP, self.folder_projVessel))

        proj_ctmip = np.clip(proj_ctmip, self.clip_min, self.clip_max)
        proj_ctmip = (proj_ctmip-self.clip_min) / (self.clip_max-self.clip_min+1e-7)

        proj_vessel = np.clip(proj_vessel, 0, 256)
        proj_vessel = (proj_vessel-0) / (256-0+1e-7)

        proj_ctmip = torch.from_numpy(proj_ctmip)[None].type(torch.float32) # (1,6*scales,256,256)
        proj_vessel = torch.from_numpy(proj_vessel)[None].type(torch.float32) # (1,6,256,256)

        'batch'
        # proj_ctmip = proj_ctmip.permute(1,0,2,3)
        proj_ctmip = proj_ctmip.reshape(-1,1,256,256)
        proj_vessel = proj_vessel.permute(1,0,2,3)

        return tuple((proj_ctmip, proj_vessel, path))


class Dataset3D_diffusion_infer(torch.utils.data.Dataset): # cropped sub volume as output
    def __init__(self, root_path, folder_projCTtopkMIP, folder_projVessel, fold_num, fold_idx, clip_min, clip_max, fixed_cases_in_trainset, name=None):
        'Initialization'
        self.name = name
        self.clip_min = 0
        self.clip_max = 400
        test_patch_path = {'projCTMIP':[], 'projVessel':[]}
        test_folder = []

        case_idx = np.arange(1,21)
        if len(fixed_cases_in_trainset)>0:
            new_list = [x for x in list(case_idx) if x not in fixed_cases_in_trainset]
            case_idx = np.array(new_list)
        random.seed(0)
        random.shuffle(case_idx)
        case_idx_split = np.split(np.array(case_idx), fold_num)
        test_folder = list(case_idx_split[fold_idx])

        for root, dirs, files in os.walk(root_path): 
            if root.split('/')[-2]!='3DircadbNifti_newLiverMask':
                continue
            if root.split('/')[-1]==folder_projCTtopkMIP: 
                for name in files:
                    if int(re.findall(r"\d+",name)[0]) in test_folder:
                        test_patch_path['projCTMIP'].append(os.path.join(root,name))

            elif root.split('/')[-1]==folder_projVessel: 
                for name in files:
                    if int(re.findall(r"\d+",name)[0]) in test_folder:
                        test_patch_path['projVessel'].append(os.path.join(root,name))

        test_patch_path['projCTMIP'] = sorted(test_patch_path['projCTMIP'])
        test_patch_path['projVessel'] = sorted(test_patch_path['projVessel'])
        self.list_IDs = test_patch_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['projCTMIP'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        proj_ctmip = nibabel.load(self.list_IDs['projCTMIP'][index]).get_fdata().transpose(2,0,1) # (6*scales,256,256)
        proj_vessel = nibabel.load(self.list_IDs['projCTMIP'][index].replace('TrainSeq_ProjCTtopkMIP_6views','TrainSeq_ProjVessel_6views')).get_fdata().transpose(2,0,1) # (6,256,256)
        path = self.list_IDs['projCTMIP'][index].split('/')[-1]

        proj_ctmip = np.clip(proj_ctmip, self.clip_min, self.clip_max)
        proj_ctmip = (proj_ctmip-self.clip_min) / (self.clip_max-self.clip_min+1e-7)

        proj_vessel = np.clip(proj_vessel, 0, 256)
        proj_vessel = (proj_vessel-0) / (256-0+1e-7)

        proj_ctmip = torch.from_numpy(proj_ctmip)[None].type(torch.float32) # (1,6*scales,256,256)
        proj_vessel = torch.from_numpy(proj_vessel)[None].type(torch.float32) # (1,6,256,256)

        'batch'
        # proj_ctmip = proj_ctmip.permute(1,0,2,3)
        proj_ctmip = proj_ctmip.reshape(-1,1,256,256)
        proj_vessel = proj_vessel.permute(1,0,2,3)

        return tuple((proj_ctmip, proj_vessel, path))
