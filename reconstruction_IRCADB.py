from scipy.fftpack import fft,ifft,fftshift,ifftshift
import ctypes as ct
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import os
import re
# import math
import nibabel
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import pickle
# import joblib
import time
# import multiprocessing
import concurrent.futures
import argparse
# import h5py
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def save_nifti(img, img_path):
    pair_img = nibabel.Nifti1Pair(img,np.eye(4))
    nibabel.save(pair_img,img_path)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class FloatBits(ct.Structure):
    _fields_ = [
        ('M', ct.c_uint, 23),
        ('E', ct.c_uint, 8),
        ('S', ct.c_uint, 1)
    ]

class Float(ct.Union):
    _anonymous_ = ('bits',)
    _fields_ = [
        ('value', ct.c_float),
        ('bits', FloatBits)
    ]

def nextpow2(x):
    if x < 0:
        x = -x
    if x == 0:
        return 0
    d = Float()
    d.value = x
    if d.M == 0:
        return d.E - 127
    return d.E - 127 + 1

def Ramp(batchSize,netChann,channNum,channSpacing): 
    N = 2**nextpow2(2*channNum-1)
    ramp = np.zeros(N)
    for i1 in range(-(channNum-1),(channNum-1)):
        if i1==0:
            ramp[i1+channNum] = 1/(4*channSpacing*channSpacing)
        elif i1 % 2 ==0:
            ramp[i1+channNum] = 0
        elif i1 % 2 ==1:
            ramp[i1+channNum] = -1/((i1*np.pi*channSpacing)**2)
    # ramp = channSpacing*np.abs(np.fft.fft(ramp))
    ramp = channSpacing*np.abs(fft(ramp))
    ramp = ramp[None,None,:,None]
    ramp = torch.from_numpy(ramp).type(torch.FloatTensor)
    ramp = ramp.repeat(batchSize,1,1,180)
    ramp = ramp.numpy()
    # ramp = fftshift(ramp)
    return ramp

def convolution(proj,batchSize,netChann,channNum,viewnum,channSpacing):
    AglPerView = np.pi/viewnum
    # channels = 512
    origin = np.zeros((batchSize,netChann,viewnum, channNum, channNum))
    # avoid truncation
    step = list(np.arange(0,1,1/100))
    step2 = step.copy()
    step2.reverse()
    step = np.array(step) # (100,)
    step = np.expand_dims(step,axis=1) # 100*1
    step = torch.from_numpy(step).type(torch.FloatTensor) # (100,1)
    step = step.repeat(batchSize,1,1,180) # 2*1*100*360
    step_temp = proj[:,:,0,:].unsqueeze(2) # 2*1*1*360
    step_temp = step_temp.repeat(1,1,100,1) # 2*1*100*360
    step = step.cuda()
    step = step*step_temp # 2*1*100*360
    step2 = np.array(step2) # (100,)
    step2 = np.expand_dims(step2,axis=1) # 100*1
    step2 = torch.from_numpy(step2).type(torch.FloatTensor) # (100,1)
    step2 = step2.repeat(batchSize,1,1,180) # 2*1*100*360
    step2_temp = proj[:,:,-1,:].unsqueeze(2) # 2*1*1*360
    step2_temp = step2_temp.repeat(1,1,100,1) # 2*1*100*360
    step2 = step2.cuda()
    step2 = step2*step2_temp # 2*1*100*360
    filterData = Ramp(batchSize,netChann,2*100+channNum,channSpacing) # 2*1*2048*360
    iLen = filterData.shape[2] # 2048
    proj = torch.cat((step,proj,step2),2) # 2*1*712*360
    proj = torch.cat((proj,torch.zeros(batchSize,netChann,iLen-proj.shape[2],viewnum).cuda()),2) # 2*1*2048*360
    sino_fft = fft(proj.detach().cpu().numpy(),axis=2) # 2*1*2048*360
    image_filter = filterData*sino_fft # 2*1*2048*360
    image_filter_ = ifft(image_filter,axis=2) # 2*1*2048*360
    image_filter_ = np.real(image_filter_)
    image_filter_ = torch.from_numpy(image_filter_).type(torch.FloatTensor)
    image_filter_final = image_filter_[:,:,100:channNum+100] # 2*1*512*360
    return image_filter_final

class reconGT_fake(nn.Module): # ground truth vessel projections are process by the pred-trained autoencoder.
    def __init__(self,first_stage_config):
        super().__init__()
        self.instantiate_first_stage(first_stage_config)
        ckpt_path = first_stage_config['params']['ckpt_path']
        self.init_from_ckpt(path=ckpt_path)

    def instantiate_first_stage(self, config): 
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval().cuda()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=True):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.first_stage_model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self, reconGT):
        posterior = self.first_stage_model.encode(reconGT) 
        z = posterior.sample()
        reconGTfake = self.first_stage_model.decode(z)
        return reconGTfake

def Vessel_Recon(sub_folder=None, key_in_config=None):
    # seq_num = 30
    fold_test_case = [['08'],['06'],['04'],['16'],['11']]

    for item in sub_folder:

        pred_root = item
        RotMat_root = "/data/3DircadbNifti_newLiverMask/TrainSeq_ProjRotMat_6views"
        # Vessel_root = "/data/3DircadbNifti_newLiverMask/Vol_resize_Vessel_256mm3"
        config_root = "/configs/ldm/vessel_seg_256_256_256"

        for root, dirs, files in os.walk(pred_root): 
            if root.split('/')[-2]!='samples':
                continue
            if 'validation_' not in root.split('/')[-1]:
                continue
            if os.path.exists(root.replace('samples','recons')):
                raise ValueError('Existed '+ root.replace('samples','recons'))
            fold_idx = int(re.findall(r"\d+",root.split('/')[-3])[-1])
            configs = [OmegaConf.load(cfg) for cfg in [os.path.join(config_root, key_in_config+'_F'+str(fold_idx)+'.yaml')]]
            config = OmegaConf.merge(*configs, {}) # all defined parameters in config.yaml
            'initialize pre-trained autoencoder'
            autoencoder = reconGT_fake(config.model.params.first_stage_config)

            for item_in_fold in fold_test_case[fold_idx]:
                # semanticVessel = nibabel.load(os.path.join(Vessel_root, 'Vessel'+item_in_fold+'.nii.gz')).get_fdata().transpose(2,0,1) # (256,256,256)
                proj_semanticVessel_list = []
                projGT_semanticVessel_list = []
                projFakeGT_semanticVessel_list = []
                inverse_rot_list = []
                # for seq_idx in range(seq_num):
                ### NOTICE!!!! SORT is needed!!!!
                for name in sorted(files):
                    if 'recon_PATIENT'+item_in_fold not in name:
                        continue
                    proj_semanticVessel = nibabel.load(os.path.join(root,name)).get_fdata().transpose(2,0,1)[:,None,] # (6,1,256,256)
                    projGT_semanticVessel = nibabel.load(os.path.join(root,name.replace('recon_','reconGT_'))).get_fdata().transpose(2,0,1)[:,None,]
                    'process ground truth vessel projections by the pre-trained autoencoder.'
                    projFakeGT_semanticVessel = autoencoder(torch.from_numpy(projGT_semanticVessel).type(torch.float32).cuda())
                    projFakeGT_semanticVessel = projFakeGT_semanticVessel.detach().cpu().numpy()
                    proj_RotMat = np.load(os.path.join(RotMat_root, name.replace('recon_','').replace('.nii.gz','.npy')), allow_pickle=True).item() 
                    inverse_rot = np.array(proj_RotMat['inverse_rot'])[:,None,] # (6,1,3,4)
                    proj_semanticVessel_list.append(proj_semanticVessel)
                    projGT_semanticVessel_list.append(projGT_semanticVessel)
                    projFakeGT_semanticVessel_list.append(projFakeGT_semanticVessel)
                    inverse_rot_list.append(inverse_rot)

                proj_semanticVessel = np.concatenate(proj_semanticVessel_list, axis=1) # (6,seq_num,256,256)
                proj_semanticVessel = proj_semanticVessel.reshape(-1,256,256) # (6*seq_num,256,256)
                projGT_semanticVessel = np.concatenate(projGT_semanticVessel_list, axis=1)
                projGT_semanticVessel = projGT_semanticVessel.reshape(-1,256,256)
                projFakeGT_semanticVessel = np.concatenate(projFakeGT_semanticVessel_list, axis=1)
                projFakeGT_semanticVessel = projFakeGT_semanticVessel.reshape(-1,256,256)
                
                'Save full-views projections'
                os.makedirs(root.replace('samples','projections'), exist_ok=True)
                save_nifti(proj_semanticVessel.transpose(1,2,0), os.path.join(root.replace('samples','projections'), 'Proj_PATIENT'+item_in_fold+'.nii.gz'))
                save_nifti(projGT_semanticVessel.transpose(1,2,0), os.path.join(root.replace('samples','projections'), 'ProjGT_PATIENT'+item_in_fold+'.nii.gz'))
                save_nifti(projFakeGT_semanticVessel.transpose(1,2,0), os.path.join(root.replace('samples','projections'), 'ProjFakeGT_PATIENT'+item_in_fold+'.nii.gz'))
                
                'proj norm'
                proj_semanticVessel[proj_semanticVessel<0] = 0
                proj_semanticVessel = proj_semanticVessel *256
                proj_semanticVessel = np.clip(proj_semanticVessel,0,256)
                projGT_semanticVessel[projGT_semanticVessel<0] = 0
                projGT_semanticVessel = projGT_semanticVessel *256
                projGT_semanticVessel = np.clip(projGT_semanticVessel,0,256)
                projFakeGT_semanticVessel[projFakeGT_semanticVessel<0] = 0
                projFakeGT_semanticVessel = projFakeGT_semanticVessel *256
                projFakeGT_semanticVessel = np.clip(projFakeGT_semanticVessel,0,256)

                inverse_rot = np.concatenate(inverse_rot_list, axis=1) # (6,seq_num,3,4)
                inverse_rot = inverse_rot.reshape(-1,3,4) # (6*seq_num,3,4)
                
                'FBP'
                filtered_proj_semanticVessel = convolution(torch.from_numpy(proj_semanticVessel.transpose(1,2,0)[:,None]).type(torch.float32).cuda(), 256, 1, 256, 180, 1)
                x = filtered_proj_semanticVessel.squeeze(1)
                x = x.permute(2,0,1)
                x = x.numpy()

                filtered_projGT_semanticVessel = convolution(torch.from_numpy(projGT_semanticVessel.transpose(1,2,0)[:,None]).type(torch.float32).cuda(), 256, 1, 256, 180, 1)
                x_GT = filtered_projGT_semanticVessel.squeeze(1)
                x_GT = x_GT.permute(2,0,1)
                x_GT = x_GT.numpy()

                filtered_projFakeGT_semanticVessel = convolution(torch.from_numpy(projFakeGT_semanticVessel.transpose(1,2,0)[:,None]).type(torch.float32).cuda(), 256, 1, 256, 180, 1)
                x_FakeGT = filtered_projFakeGT_semanticVessel.squeeze(1)
                x_FakeGT = x_FakeGT.permute(2,0,1)
                x_FakeGT = x_FakeGT.numpy()

                theta = torch.from_numpy(inverse_rot)[None].type(torch.float32) # (1,6*2,3,4)

                'sampled projection reconstruction'
                x_repeat = torch.from_numpy(x[:,:,:,None]).type(torch.float32).repeat(1,1,1,256) # (6,256,256,256)
                x_repeat = x_repeat.unsqueeze(1) # (6,1,256,256,256)
                grid = F.affine_grid(theta.reshape(-1,3,4), x_repeat.size()) # (6,256,256,256,3)
                x_rotate = F.grid_sample(x_repeat, grid) # (6,1,256,256,256)
                x_rotate = torch.sum(x_rotate, dim=0) # (1,256,256,256)

                'ground truth projection reconstruction'
                x_GT_repeat = torch.from_numpy(x_GT[:,:,:,None]).type(torch.float32).repeat(1,1,1,256) # (6,256,256,256)
                x_GT_repeat = x_GT_repeat.unsqueeze(1) # (6,1,256,256,256)
                grid = F.affine_grid(theta.reshape(-1,3,4), x_GT_repeat.size()) # (6,256,256,256,3)
                x_GT_rotate = F.grid_sample(x_GT_repeat, grid) # (6,1,256,256,256)
                x_GT_rotate = torch.sum(x_GT_rotate, dim=0) # (1,256,256,256)

                'fake ground truth projection reconstruction'
                x_FakeGT_repeat = torch.from_numpy(x_FakeGT[:,:,:,None]).type(torch.float32).repeat(1,1,1,256) # (6,256,256,256)
                x_FakeGT_repeat = x_FakeGT_repeat.unsqueeze(1) # (6,1,256,256,256)
                grid = F.affine_grid(theta.reshape(-1,3,4), x_FakeGT_repeat.size()) # (6,256,256,256,3)
                x_FakeGT_rotate = F.grid_sample(x_FakeGT_repeat, grid) # (6,1,256,256,256)
                x_FakeGT_rotate = torch.sum(x_FakeGT_rotate, dim=0) # (1,256,256,256)

                os.makedirs(root.replace('samples','recons'), exist_ok=True)
                save_nifti(x_rotate.squeeze(0).numpy().transpose(2,1,0), os.path.join(root.replace('samples','recons'), 'ReconFBP_PATIENT'+item_in_fold+'.nii.gz'))
                save_nifti(x_GT_rotate.squeeze(0).numpy().transpose(2,1,0), os.path.join(root.replace('samples','recons'), 'ReconGTFBP_PATIENT'+item_in_fold+'.nii.gz'))
                save_nifti(x_FakeGT_rotate.squeeze(0).numpy().transpose(2,1,0), os.path.join(root.replace('samples','recons'), 'ReconFakeGTFBP_PATIENT'+item_in_fold+'.nii.gz'))


def load_submatrix(file_path):
    """Function to load a single submatrix."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def parallel_load_and_concat(filepaths, num_processes):
    """Load submatrices in parallel and concatenate them."""
    submatrices_parallel = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(load_submatrix, filepaths))
        submatrices_parallel.extend(results)
    return np.concatenate(submatrices_parallel, axis=0)

def prox_solver(ipt=None, sino_sample=None, rho=None, solver_mode='apgm', device=None, sub_folder=None):
    """
    This code solves the following optimization problem:
    argmin_x ||Ax-b||_2^2 + ||x-u||^2_2       
    return: image x, (N, C, H, W)
    """
    def grad(x, sino, A, At):
        """
        performs gradient of ||Ax-b||_2^2
        return: gradient wrt image x, (N, C, H, W)
        """
        # grad = self.ftran(self.fmult(x) - sino)
        diff_proj = []
        diff_proj_filtered = []
        for cnt, A_sub in enumerate(A): # (view1,..., view180)
            diff = A_sub.dot(x.ravel()).reshape(256,256) - sino[:,:,cnt]
            diff_proj.append(diff)
        diff_proj = np.stack(diff_proj, axis=-1)
        diff_proj_filtered = convolution(torch.from_numpy(diff_proj[:,None,]).type(torch.float32).cuda(), 256, 1, 256, 180, 1)
        diff_proj_filtered = diff_proj_filtered.squeeze(1).cpu().numpy()
        diff_proj_filtered = [diff_proj_filtered[:,:,i] for i in range(180)]
        
        diff_vol = np.zeros(256**3)  # Initialize the back-projected volume
        for back_proj_matrix, projection in zip(At, diff_proj_filtered):
            diff_vol += back_proj_matrix @ projection.ravel()
        # Reshape the back-projected volume to its original 3D shape
        grad_ = diff_vol.reshape((256, 256, 256))
        grad_ = grad_ /256
        return grad_

    def apgm(ipt, sino_sample, rho, A, At, mask=None, save_root=None, case_idx=None, device=None):
        """
        performs APGM algorithm
        return: image x, (N, C, H, W)
        """
        accelerate=False #True
        # x = ipt["sample_data"].detach().cpu().clone().to(device)
        # z = ipt["sample_data"].detach().cpu().clone().to(device)
        x = ipt.copy()
        z = ipt.copy()
        s = x
        # t = torch.tensor([1.]).float().to(device)
        # t = t.cpu().numpy()
        x_opt_list = []
        for idx in range(10):
            print('iter_'+str(idx))
            # percentile_95 = np.percentile(s.flatten(), 95)
            # s_clip = np.clip(s, percentile_95, 1)
            grad_ = grad(s, sino_sample, A, At)
            # grad_ = grad_ * mask
            xnext = s - 1e-1*grad_ #- rho*(s - z) - 1e-1*tv(s_clip)
            # acceleration
            # if accelerate:
            #     tnext = 0.5*(1+torch.sqrt(1+4*t*t))
            # else:
            #     tnext = 1
            # s = xnext + ((t-1)/tnext)*(xnext-x)
            s = xnext
            # update
            # t = tnext
            x = xnext
            # x = x * mask
            x_opt_list.append(x*mask)
            save_nifti(x*mask, os.path.join(save_root, 'ReconFBP_PATIENT'+case_idx+'_optIter_'+str(idx)+'.nii.gz'))
        # return x, x_opt_list  


    'Data preparation'
    for item in sub_folder:
        pred_root = item.replace('samples','projections')
        ct_path = "/data/3DircadbNifti_newLiverMask/Vol_resize_CT_256mm3"
        liver_path = "/data/3DircadbNifti_newLiverMask/Vol_resize_liver_256mm3"
        # vessel_path = "/data/3DircadbNifti_newLiverMask/Vol_resize_semanticVessel_256mm3"
        vol_size = 256
        fold_test_case = [['08'],['06'],['04'],['16'],['11']]

        save_folder = 'recons_singleTermOpt'

        num_splits = 30
        num_processes = 8
        base_path = '/system_matrix'
        
        filepaths_forward = [os.path.join(base_path, f'sub_sparse_matrices_forward_{idx}.pkl') 
                            for idx in range(num_splits)]
        filepaths_back = [os.path.join(base_path, f'sub_sparse_matrices_back_{idx}.pkl') 
                        for idx in range(num_splits)]
        
        # Load forward matrix
        start_time = time.time()
        system_matrices = parallel_load_and_concat(filepaths_forward, num_processes)
        parallel_time = time.time() - start_time
        print(f"Parallel loading time for forward matrix: {parallel_time:.2f} seconds")
        
        # Load back matrix
        start_time = time.time()
        system_matrices_back = parallel_load_and_concat(filepaths_back, num_processes)
        parallel_time = time.time() - start_time
        print(f"Parallel loading time for back matrix: {parallel_time:.2f} seconds")

        for root, dirs, files in os.walk(pred_root): 
            if root.split('/')[-2]!='projections':
                continue
            if 'validation_' not in root.split('/')[-1]:
                continue
            fold_idx = int(re.findall(r"\d+",root.split('/')[-3])[-1])
            if os.path.exists(root.replace('projections',save_folder)):
                continue

            for item_in_fold in fold_test_case[fold_idx]:
                for name in sorted(files): ## NOTICE!!!! Files must be sorted!!!!
                    if 'Proj_PATIENT'+item_in_fold not in name:
                        continue
                    liver_mask = nibabel.load(os.path.join(liver_path, 'liver'+item_in_fold+'.nii.gz')).get_fdata()
                    liver_mask[liver_mask>0] = 1
                    b = nibabel.load(os.path.join(root,name)).get_fdata() # generated projections
                    b = b * 256
                    filtered_b = convolution(torch.from_numpy(b[:,None]).type(torch.float32).cuda(), 256, 1, 256, 180, 1)
                    filtered_b = filtered_b.squeeze(1).cpu().numpy()
                    u = nibabel.load(os.path.join(ct_path, 'PATIENT'+item_in_fold+'.nii.gz')).get_fdata() # CT volume
                    u = np.clip(u,0,400)
                    u = (u-0) / (400-0)
                    x = u # initialize x with CT images

                    '# Perform the back-projection by iterating through each view'
                    projections = [filtered_b[:,:,i] for i in range(180)]
                    vol_back_projected = np.zeros(vol_size**3)  # Initialize the back-projected volume
                    for back_proj_matrix, projection in zip(system_matrices_back, projections):
                        vol_back_projected += back_proj_matrix @ projection.ravel()
                    # Reshape the back-projected volume to its original 3D shape
                    vol_back_projected = vol_back_projected.reshape((vol_size, vol_size, vol_size))
                    save_path_1 = os.path.join(root.replace('projections',save_folder), 'ReconFBP_PATIENT'+item_in_fold+'.nii.gz')
                    os.makedirs(root.replace('projections',save_folder), exist_ok=True)
                    save_nifti(vol_back_projected, save_path_1)
                    
                    if solver_mode=='apgm':
                        rho = 0#1e-2
                        apgm(ipt=u, sino_sample=b, rho=rho, A=system_matrices, At=system_matrices_back, mask=liver_mask, 
                                            save_root=root.replace('projections',save_folder), case_idx=item_in_fold)


def Binary_Segments(sub_folder=None):
    for item in sub_folder:
        pred_root = item.replace('samples','recons_singleTermOpt')
        for root, dirs, files in os.walk(pred_root): 
            if root.split('/')[-2]!='recons_singleTermOpt':
                continue
            if 'validation_iter' not in root.split('/')[-1]:
                continue
            for name in sorted(files):
                if 'optIter' not in name:
                    continue

                recon_vessel = nibabel.load(os.path.join(root, name)).get_fdata()
                flattened_data = recon_vessel.flatten()
                flattened_data = flattened_data[flattened_data > 0]
                # Calculate the 95th percentile
                percentile_95 = np.percentile(flattened_data, 95)

                # # Plot the histogram
                # plt.hist(flattened_data, bins=50, edgecolor='black')
                # # Add a vertical line at the 95th percentile
                # plt.axvline(percentile_95, color='r', linestyle='dashed', linewidth=2)
                # # Add title and labels
                # plt.title(f'Histogram with 95th Percentile at {percentile_95:.2f}')
                # plt.xlabel('Value')
                # plt.ylabel('Frequency')
                # # Display the plot
                # plt.show()

                # recon_vessel = np.clip(recon_vessel, 0.3, 1) # patient04
                # recon_vessel = np.clip(recon_vessel, 0.2, 1) # patient16
                recon_vessel = np.clip(recon_vessel, percentile_95, 1)
                recon_vessel = (recon_vessel-recon_vessel.min()) / (recon_vessel.max()-recon_vessel.min())
                recon_vessel[recon_vessel>0] = 1
                # recon_vessel[recon_vessel<0.1] = 0
                save_nifti(recon_vessel, os.path.join(root,name.replace('ReconFBP','BinaryReconFBP')))


def optimized_forward_projection(sub_folder=None):
    for item in sub_folder:
        vol_size = 256
        save_folder = 'projections_singleTermOpt'
        fold_test_case = [['08'],['06'],['04'],['16'],['11']]
        pred_root = item.replace('samples','recons_singleTermOpt')
        num_splits = 30
        num_processes = 8
        base_path = '/system_matrix'
        
        filepaths_forward = [os.path.join(base_path, f'sub_sparse_matrices_forward_{idx}.pkl') 
                            for idx in range(num_splits)]
        
        # Load forward matrix
        start_time = time.time()
        system_matrices = parallel_load_and_concat(filepaths_forward, num_processes)
        parallel_time = time.time() - start_time
        print(f"Parallel loading time for forward matrix: {parallel_time:.2f} seconds")
        
        for root, dirs, files in os.walk(pred_root): 
            if root.split('/')[-2]!='recons_singleTermOpt':
                continue
            if 'validation_' not in root.split('/')[-1]:
                continue
            fold_idx = int(re.findall(r"\d+",root.split('/')[-3])[-1])
            if os.path.exists(root.replace('recons_singleTermOpt',save_folder)):
                continue
                # raise ValueError('Existed ' + root.replace('recons_singleTermOpt', save_folder))

            for item_in_fold in fold_test_case[fold_idx]:
                for name in sorted(files): ## NOTICE!!!! Files must be sorted!!!!
                    if 'optIter' not in name:
                        continue
                    if 'BinaryReconFBP' not in name:
                        continue
                    BinaryReconFBP_opt = nibabel.load(os.path.join(root,name)).get_fdata()
                    ReconFBP_opt = nibabel.load(os.path.join(root,name.replace('BinaryReconFBP','ReconFBP'))).get_fdata()
                    BinaryReconFBP_opt = (BinaryReconFBP_opt-BinaryReconFBP_opt.min()) / (BinaryReconFBP_opt.max()-BinaryReconFBP_opt.min())
                    ReconFBP_opt = (ReconFBP_opt-ReconFBP_opt.min()) / (ReconFBP_opt.max()-ReconFBP_opt.min())

                    proj_BinaryReconFBP_opt = []
                    proj_ReconFBP_opt = []
                    for cnt, A_sub in enumerate(system_matrices): # (view1,..., view180)
                        temp = A_sub.dot(BinaryReconFBP_opt.ravel()).reshape(256,256) 
                        proj_BinaryReconFBP_opt.append(temp)
                        temp = A_sub.dot(ReconFBP_opt.ravel()).reshape(256,256) 
                        proj_ReconFBP_opt.append(temp)
                    proj_BinaryReconFBP_opt = np.stack(proj_BinaryReconFBP_opt, axis=-1)
                    proj_ReconFBP_opt = np.stack(proj_ReconFBP_opt, axis=-1)
                    
                    os.makedirs(root.replace('recons_singleTermOpt',save_folder), exist_ok=True)
                    save_path_1 = os.path.join(root.replace('recons_singleTermOpt',save_folder), 'ProjOf'+name)
                    save_path_2 = os.path.join(root.replace('recons_singleTermOpt',save_folder), 'ProjOf'+name.replace('BinaryReconFBP','ReconFBP'))
                    save_nifti(proj_BinaryReconFBP_opt, save_path_1)
                    save_nifti(proj_ReconFBP_opt, save_path_2)


def noise_remove_connected_region_2(sub_folder=None): # skimage.measure.label & scipy.ndimage.label
    from skimage import morphology, measure
    from scipy.ndimage import label, generate_binary_structure
    def find_outliers_iqr(data):
        # Calculate the interquartile range
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr_value = q3 - q1
        # Define the lower and upper bounds for outliers
        lower_bound = q1 - 1.5 * iqr_value
        upper_bound = q3 + 1.5 * iqr_value
        # Identify outliers
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        return outliers
    
    # Read the segmented 3D binary volume
    for item in sub_folder:
        pred_root = item.replace('samples','recons_singleTermOpt')
        connectivity=2
        for root, dirs, files in os.walk(pred_root): 
            for name in files:
                if os.path.exists(os.path.join(root, name.replace('BinaryReconFBP','noiseCancelConnect'))):
                    continue
                if 'BinaryReconFBP' not in name:
                    continue
                if 'optIter_4' not in name:
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

def cases_split(N_idx, N, key_in_pred_folder, key_in_config):
    data_root = '/model/latent_diffusion/outputs'
    data_path_list = []
    for root, dirs, files in os.walk(data_root): 
        if key_in_pred_folder not in root.split('/')[-3]: 
            continue
        if root.split('/')[-2] != 'samples':
            continue
        if 'validation_iter' not in root.split('/')[-1]:
            continue
        if root not in data_path_list:
            data_path_list.append(root)
        
    data_path_list = sorted(data_path_list)

    split_data = split_list_ordered(data_path_list, N)

    # Print each part with element names
    print(f"Part {N_idx}:")
    for element in split_data[N_idx]:
        print(f" - {element}")
    
    ## generated projections (artifacts appear on projections) -> recontructed vessel tree 
    Vessel_Recon(sub_folder=split_data[N_idx], key_in_config=key_in_config) 

    ## artifacts suppresion -> optimized vessel tree
    prox_solver(sub_folder=split_data[N_idx]) # 

    ## optimized vessel tree -> binarized vessel tree 
    Binary_Segments(sub_folder=split_data[N_idx])

    ## check projections of optimized vessel tree and binarized vessel tree
    optimized_forward_projection(sub_folder=split_data[N_idx])


def main(args):
    cases_split(N_idx=args.N_idx, N=args.N, 
                key_in_pred_folder=args.key_in_pred_folder,
                key_in_config=args.key_in_config)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='temp')
    parser.add_argument('--N', default=1, type=int)
    parser.add_argument('--N_idx', default=-1, type=int)
    parser.add_argument('--key_in_pred_folder', type=str)
    parser.add_argument('--key_in_config', type=str)
    args = parser.parse_args()

    main(args)

