import numpy as np
import math
import nibabel
import os
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
import SimpleITK as sitk
from scipy.interpolate import interpn

def save_nifti(img, img_path):
    pair_img = nibabel.Nifti1Pair(img,np.eye(4))
    nibabel.save(pair_img,img_path)

def CT_resize(): # crop to the liver boundary and resize to 256  (512*512*D) -> (256*256*256) For training set
    root_path = r'/data/3DircadbNifti_newLiverMask'
    for root, dirs, files in os.walk(root_path): 
        if root.split('/')[-1]!='patientCTVolume':
            continue
        for name in files:
            ct_vol = sitk.ReadImage(os.path.join(root,name))
            origin = ct_vol.GetOrigin()
            space = ct_vol.GetSpacing()
            direction = ct_vol.GetDirection()

            patient_idx = re.findall(r"\d+",name)
            liver_mask = sitk.ReadImage(os.path.join(root.replace('patientCTVolume', 'liver'), 'liver'+patient_idx[-1]+'.nii.gz'))
            hepatic_tree = sitk.ReadImage(os.path.join(root.replace('patientCTVolume', 'hepaticVein'), 'hepatic'+patient_idx[-1]+'.nii.gz'))
            portal_tree = sitk.ReadImage(os.path.join(root.replace('patientCTVolume', 'portalVein'), 'portal'+patient_idx[-1]+'.nii.gz'))

            ct_vol = sitk.GetArrayFromImage(ct_vol).transpose(1,2,0)+1024
            liver_mask = sitk.GetArrayFromImage(liver_mask).transpose(1,2,0)
            hepatic_tree = sitk.GetArrayFromImage(hepatic_tree).transpose(1,2,0)
            portal_tree = sitk.GetArrayFromImage(portal_tree).transpose(1,2,0)

            liver_dist = liver_mask
            liver_dist[liver_dist>0] = 1
            edge = np.argwhere(liver_dist==1)
            row_min = edge[:,0].min()
            row_max = edge[:,0].max()
            col_min = edge[:,1].min()
            col_max = edge[:,1].max()
            depth_min = edge[:,2].min()
            depth_max = edge[:,2].max()

            'crop CT image and liver vessel mask to liver region as ROI'
            cropped_ct = ct_vol[row_min:row_max, col_min:col_max, depth_min:depth_max]
            cropped_hepatic = hepatic_tree[row_min:row_max,col_min:col_max,depth_min:depth_max]
            cropped_portal = portal_tree[row_min:row_max,col_min:col_max,depth_min:depth_max]
            cropped_liver = liver_mask[row_min:row_max,col_min:col_max,depth_min:depth_max]

            H,W,D = cropped_ct.shape
            x = np.linspace(0, H-1, H).astype(int)
            y = np.linspace(0, W-1, W).astype(int)
            z = np.linspace(0, D-1, D).astype(int)
            points = (x,y,z)
            xx = np.linspace(0, H-1, 256) 
            yy = np.linspace(0, W-1, 256)
            zz = np.linspace(0, D-1, 256)
            xx,yy,zz = np.meshgrid(xx, yy, zz)
            xx,yy,zz = xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)
            # vol = np.concatenate((xx[:,None], yy[:,None], zz[:,None]), axis=-1)
            points_interp = (xx,yy,zz)

            space_resize = ((space[0]*H)/256, (space[1]*W)/256, (space[2]*D)/256)
            # vol = value_func_3d(*np.meshgrid(*points, indexing='ij'))
            vol_resize = interpn(points, cropped_ct, points_interp, method='linear')
            vol_resize = vol_resize.reshape(256,256,256).transpose(1,0,2)

            hepatic_resize = interpn(points, cropped_hepatic, points_interp, method='nearest')
            hepatic_resize = hepatic_resize.reshape(256,256,256).transpose(1,0,2)
            portal_resize = interpn(points, cropped_portal, points_interp, method='nearest')
            portal_resize = portal_resize.reshape(256,256,256).transpose(1,0,2)
            liver_resize = interpn(points, cropped_liver, points_interp, method='nearest')
            liver_resize = liver_resize.reshape(256,256,256).transpose(1,0,2)

            # continuous_mask = liver_resize + hepatic_resize + portal_resize
            # continuous_mask[continuous_mask>0]  =1
            vol_resize = vol_resize*liver_resize
            hepatic_resize = hepatic_resize*liver_resize
            portal_resize = portal_resize*liver_resize

            hepatic_resize[hepatic_resize>0] = 1
            portal_resize[portal_resize>0] = 1

            # vessel_resize = hepatic_resize + portal_resize
            # vessel_resize[vessel_resize>0] = 1

            save_path_1 = os.path.join(root_path, 'Vol_resize_CT_256mm3', 'PATIENT'+patient_idx[-1]+'.nii.gz')
            save_path_2 = os.path.join(root_path, 'Vol_resize_hepatic_256mm3', 'hepatic'+patient_idx[-1]+'.nii.gz')
            save_path_3 = os.path.join(root_path, 'Vol_resize_portal_256mm3', 'portal'+patient_idx[-1]+'.nii.gz')
            save_path_4 = os.path.join(root_path, 'Vol_resize_liver_256mm3', 'liver'+patient_idx[-1]+'.nii.gz')
            # save_path_5 = os.path.join(root_path, 'Vol_resize_Vessel_256mm3', 'Vessel'+patient_idx[-1]+'.nii.gz')

            os.makedirs(os.path.join(root_path, 'Vol_resize_CT_256mm3'), exist_ok=True)
            os.makedirs(os.path.join(root_path, 'Vol_resize_hepatic_256mm3'), exist_ok=True)
            os.makedirs(os.path.join(root_path, 'Vol_resize_portal_256mm3'), exist_ok=True)
            os.makedirs(os.path.join(root_path, 'Vol_resize_liver_256mm3'), exist_ok=True)
            # os.makedirs(os.path.join(root_path, 'Vol_resize_Vessel_256mm3'), exist_ok=True)

            img = sitk.GetImageFromArray(vol_resize.transpose(2,0,1)-1024)
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(space_resize)
            sitk.WriteImage(img, '{}'.format(save_path_1))

            img = sitk.GetImageFromArray(hepatic_resize.transpose(2,0,1))
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(space_resize)
            sitk.WriteImage(img, '{}'.format(save_path_2))

            img = sitk.GetImageFromArray(portal_resize.transpose(2,0,1))
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(space_resize)
            sitk.WriteImage(img, '{}'.format(save_path_3))

            img = sitk.GetImageFromArray(liver_resize.transpose(2,0,1))
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(space_resize)
            sitk.WriteImage(img, '{}'.format(save_path_4))

            # img = sitk.GetImageFromArray(vessel_resize.transpose(2,0,1))
            # img.SetOrigin(origin)
            # img.SetDirection(direction)
            # img.SetSpacing(space_resize)
            # sitk.WriteImage(img, '{}'.format(save_path_5))

class Rot3dImg(nn.Module):
    def __init__(self, ite_views=1, views=180, start_view=0, batchSize=1):
        super(Rot3dImg, self).__init__()
        self.views = views
        self.batchSize = batchSize
        self.start_view = start_view
        self.ite_views = ite_views

    def forward(self, x):
        '''
            x: image 
            x is a tensor (B,1,D,H,W)==(B,1,256,256,256)
        '''
        ''' rotate'''
        unit_interval = math.pi/self.views
        min_unit_interval = math.pi/180
        x_rotate_allViews = []
        batchSize = self.batchSize * x.shape[1] # B*classes
        x = x.reshape(batchSize,1,x.shape[-3],x.shape[-2],x.shape[-1])
        # if self.start_view==0:
        #     return x.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy(), None
        for i in range(self.ite_views):
            angle = self.start_view * min_unit_interval + i * unit_interval
            theta = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                                [np.sin(angle), np.cos(angle), 0, 0],
                                [0, 0, 1, 0]])                                 
            theta = torch.from_numpy(theta).type(torch.FloatTensor)
            theta = theta.repeat(batchSize,1,1)
            theta = theta.cuda()
            ''' interpolation'''
            grid = F.affine_grid(theta, x.size())
            x_rotate = F.grid_sample(x, grid) # 4*1*256*256*256 (B,1,D,H,W)
            x_rotate_allViews.append(x_rotate)
        x_rotate_allViews = torch.cat(x_rotate_allViews,dim=1) # (B,V,D,H,W)
        x_rotate_allViews = x_rotate_allViews.permute(1,0,2,3,4) # (V,B,D,H,W)
        x_rotate_allViews = x_rotate_allViews.squeeze(dim=0).detach().cpu().numpy() # (B,D,H,W)
        return x_rotate_allViews, theta.cpu().numpy()

def k_largest_values_and_indices(image, axis, k=32):
    # Swap the target axis with the last axis to use torch.topk()
    image = image.permute(*[dim for dim in range(image.dim()) if dim != axis], axis)

    # Reshape the tensor to make the axis to be reduced the last dimension
    shape = image.shape
    reshaped_image = image.reshape(-1, shape[-1])

    # Find the k largest values and their indices along the last dimension
    top_values, top_indices = torch.topk(reshaped_image, k=k, dim=-1)

    # Reshape the top_values tensor back to the original shape
    top_values = top_values.view(*shape[:-1], k)

    top_indices = top_indices.view(*shape[:-1], k)


    return top_values, top_indices

class FP(nn.Module):
    def __init__(self, views, detH, detW, start_view=0, batchSize=1, mode=None):
        super(FP, self).__init__()
        self.views = views
        self.detH = detH
        self.detW = detW
        self.batchSize = batchSize
        # self.start_angle = start_angle # radian
        self.start_view = start_view
        self.mode = mode

    def forward(self, x, k=32):
        '''
            x: image 
            x is a tensor (B,1,D,H,W)==(B,1,256,256,256)
        '''
        if self.mode=='topkMIP':
            sino = torch.from_numpy( np.zeros((self.batchSize, self.detH, self.detW, k, self.views))).type(torch.FloatTensor) # (B,detH,detD,S,V)
            mip_index = torch.from_numpy( np.zeros((self.batchSize, self.detH, self.detW, k, self.views))).type(torch.FloatTensor) # (B,detH,detD,S,V)
        elif self.mode=='MIP':
            sino = torch.from_numpy( np.zeros((self.batchSize, self.detH, self.detW, self.views))).type(torch.FloatTensor) # (B,detH,detD,V)
            mip_index = torch.from_numpy( np.zeros((self.batchSize, self.detH, self.detW, self.views))).type(torch.FloatTensor) # (B,detH,detD,V)
        else:
            batchSize = self.batchSize * x.shape[1]
            x = x.reshape(-1,1,x.shape[-3],x.shape[-2],x.shape[-1])
            sino = torch.from_numpy( np.zeros((batchSize, self.detH, self.detW, self.views))).type(torch.FloatTensor) # (B,detH,detD,V)
        sino = sino.cuda()
        ''' rotate'''
        unit_interval = math.pi/self.views
        min_unit_interval = math.pi/180
        for i in range(self.views):
            if self.start_view==0:
                x_rotate = x
            else:
                # angle = self.start_angle + i * unit_interval
                angle = self.start_view * min_unit_interval + i * unit_interval
                theta = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                                    [np.sin(angle), np.cos(angle), 0, 0],
                                    [0, 0, 1, 0]])                                 
                theta = torch.from_numpy(theta).type(torch.FloatTensor)
                theta = theta.repeat(batchSize,1,1)
                theta = theta.cuda()
                ''' interpolation'''
                grid = F.affine_grid(theta, x.size())
                x_rotate = F.grid_sample(x, grid) # 1*1*256*256*256 (B,1,D,H,W)

            if self.mode=='MIP':
                'maximum'
                sino[:,:,:,i] = torch.max(x_rotate, dim=-1)[0].squeeze(dim=1) 
                mip_index[:,:,:,i] = torch.max(torch.clamp(x_rotate, min=0, max=400), dim=-1)[1].squeeze(dim=1)  # index on dim W. index range is [0,256)
            elif self.mode=='topkMIP':
                # sino[:,:,:,:,i] = k_largest_values_and_indices(x_rotate, 4, k=k)[0].squeeze(dim=1) 
                # mip_index[:,:,:,:,i] = k_largest_values_and_indices(torch.clamp(x_rotate, min=0, max=400), 4, k=k)[1].squeeze(dim=1)
                sino[:,:,:,:,i] = k_largest_values_and_indices(x_rotate, 5, k=k)[0].squeeze(dim=1) 
                mip_index[:,:,:,:,i] = k_largest_values_and_indices(torch.clamp(x_rotate, min=0, max=400), 5, k=k)[1].squeeze(dim=1)
            else:
                ''' accumulation'''
                sino[:,:,:,i] = torch.sum(x_rotate, dim=-1).squeeze(dim=1) # sino: (B,D,W,V)
        sino = sino.squeeze(dim=0).detach().cpu().numpy() # 256*256*180
        try:
            mip_index = mip_index.squeeze(dim=0).detach().cpu().numpy() # 256*256*32*180
            return sino, mip_index
        except:
            return sino

def projections_of_3DImg(hepatic_mask_path=None, portal_mask_path=None, ct_path=None, views=180):

    hepatic_mask = nibabel.load(hepatic_mask_path).get_fdata()
    portal_mask = nibabel.load(portal_mask_path).get_fdata()
    ct_vol = nibabel.load(ct_path).get_fdata()
    case_id = hepatic_mask_path.split('/')[-1].split('.')[0].replace('hepatic','PATIENT')

    ite_views = 1 # Rotate only one time for each call for Rot3dImg()
    fp = FP(views=ite_views,detH=256,detW=256,start_view=0,batchSize=1)
    fptopkMIP = FP(views=ite_views,detH=256,detW=256,start_view=0,batchSize=1,mode='topkMIP')
    fpMIP = FP(views=ite_views,detH=256,detW=256,start_view=0,batchSize=1,mode='MIP')

    data_root = r'/data/3DircadbNifti_newLiverMask'

    for v_i in range(views):
        'Rotate the vessel masks based on the projection view'
        rotCT = Rot3dImg(ite_views=ite_views, views=views, start_view=v_i)

        hepatic_mask_rot = rotCT(torch.from_numpy(hepatic_mask.transpose(2,0,1)[None,None,]).type(torch.FloatTensor).cuda())[0]
        portal_mask_rot = rotCT(torch.from_numpy(portal_mask.transpose(2,0,1)[None,None,]).type(torch.FloatTensor).cuda())[0]
        ct_vol_rot = rotCT(torch.from_numpy(ct_vol.transpose(2,0,1)[None,None,]).type(torch.FloatTensor).cuda())[0]

        ###################################################################
        hepatic_mask_rot[hepatic_mask_rot>0] = 1
        portal_mask_rot[portal_mask_rot>0] = 1
        hepatic_mask_rot = hepatic_mask_rot.astype(int)
        portal_mask_rot = portal_mask_rot.astype(int)
        mask_rot = hepatic_mask_rot + portal_mask_rot
        mask_rot[mask_rot>1] = 1
        #####################################################################

        ct_mip_proj, _ = fpMIP(torch.from_numpy(ct_vol_rot[None,None,]).type(torch.FloatTensor).cuda())
        ct_mip_proj[ct_mip_proj<0] = 0

        ct_topkmip_proj, _ = fptopkMIP(torch.from_numpy(ct_vol_rot[None,None,]).type(torch.FloatTensor).cuda())
        ct_topkmip_proj = ct_topkmip_proj.squeeze(-1)
        ct_topkmip_proj[ct_topkmip_proj<0] = 0
        
        mask_rot_binary = mask_rot.copy()
        mask_rot_binary[mask_rot_binary>0] = 1
        vessel_proj = fp(torch.from_numpy(mask_rot_binary[None,None,]).type(torch.FloatTensor).cuda())

        os.makedirs(os.path.join(data_root,'ProjCTMIP_256_256_1',case_id), exist_ok=True)
        os.makedirs(os.path.join(data_root,'ProjCTtopkMIP_256_256_32',case_id), exist_ok=True)
        os.makedirs(os.path.join(data_root,'ProjVessel_256_256_1',case_id), exist_ok=True)

        save_nifti(ct_mip_proj, os.path.join(data_root,'ProjCTMIP_256_256_1',case_id,'View_'+str(v_i)+'_Proj_CTMIP.nii.gz'))
        save_nifti(ct_topkmip_proj, os.path.join(data_root,'ProjCTtopkMIP_256_256_32',case_id,'View_'+str(v_i)+'_Proj_CTMIP.nii.gz'))
        save_nifti(vessel_proj, os.path.join(data_root,'ProjVessel_256_256_1',case_id,'View_'+str(v_i)+'_Proj_Vessel.nii.gz'))

def projections():
    views = 180 # half circle
    # views = 360 # full circle
    data_root = r'/data/3DircadbNifti_newLiverMask'

    for root, dirs, files in os.walk(data_root): 
        if root.split('/')[-1]!='Vol_resize_CT_256mm3': 
            continue
        for name in files:
            ct_path = os.path.join(root,name)
            hepatic_mask_path = os.path.join(root.replace('Vol_resize_CT_256mm3','Vol_resize_hepatic_256mm3'), name.replace('PATIENT','hepatic'))
            portal_mask_path = os.path.join(root.replace('Vol_resize_CT_256mm3', 'Vol_resize_portal_256mm3'), name.replace('PATIENT','portal'))

            projections_of_3DImg(hepatic_mask_path=hepatic_mask_path, 
                                 portal_mask_path=portal_mask_path, 
                                 ct_path=ct_path, 
                                 views=views)

def cat_projections_batch(): # concatenate the projections
    # Custom sorting function
    def sort_key(path):
        # Extract the number from the 'View_xx_Proj_CTMIP' string
        view_number = int(path.split('View_')[1].split('_')[0])
        return view_number

    def inverse_rot_matrix(current_view, origin_view=0, views=180):
        unit_interval = math.pi/views
        angle = current_view * unit_interval - origin_view * unit_interval
        theta = np.array([[np.cos(angle), np.sin(angle), 0, 0],
                          [-np.sin(angle), np.cos(angle), 0, 0],
                          [0, 0, 1, 0]] )
        return theta 

    def forward_rot_matrix(current_view, origin_view=0, views=180):
        unit_interval = math.pi/views
        angle = current_view * unit_interval - origin_view * unit_interval
        theta = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                            [np.sin(angle), np.cos(angle), 0, 0],
                            [0, 0, 1, 0]])    
        # theta = np.array([[np.cos(angle), -np.sin(angle), np.zeros(angle.shape), np.zeros(angle.shape)],
        #                     [np.sin(angle), np.cos(angle), np.zeros(angle.shape), np.zeros(angle.shape)],
        #                     [np.zeros(angle.shape), np.zeros(angle.shape), np.ones(angle.shape), np.zeros(angle.shape)]])   
        return theta 

    data_root = r'/data/3DircadbNifti_newLiverMask/ProjCTMIP_256_256_1'
    interval = 30 # degs   6 views per sequence
    views = 180 # degs half-circle for parallel beam
    sequence_num = int(views/interval)
    path_groups = {}
    for root, dirs, files in os.walk(data_root): 
        for name in files:
            patient_idx = root.split('/')[-1]
            sequence_idx = str(int(int(re.findall(r'\d+',name)[0]) % interval)).zfill(2)
            if patient_idx not in path_groups.keys():
                path_groups.update({patient_idx:{}})
            if sequence_idx not in path_groups[patient_idx].keys():
                path_groups[patient_idx].update({sequence_idx:[]}) 
            path_groups[patient_idx][sequence_idx].append(os.path.join(root,name))

    for patient_idx in sorted(list(path_groups.keys())):
        for sequence_idx in sorted(list(path_groups[patient_idx].keys())): # 30 sequences
            ct_topkmip_proj_list = []
            vessel_proj_list = []
            sorted_paths = sorted(path_groups[patient_idx][sequence_idx], key=sort_key)
            inverse_theta_list = []
            forward_theta_list = []
            rot_mat = {'inverse_rot':[],'forward_rot':[]}
            for inner_seq_idx in range(len(sorted_paths)): # 6 views per sequence
                ct_topkmip_proj = nibabel.load(sorted_paths[inner_seq_idx]
                                        .replace('ProjCTMIP_256_256_1', 'ProjCTtopkMIP_256_256_32')).get_fdata()
                ct_topkmip_proj[ct_topkmip_proj<0] = 0
                vessel_proj = nibabel.load(sorted_paths[inner_seq_idx]
                                              .replace('ProjCTMIP_256_256_1','ProjVessel_256_256_1')
                                              .replace('Proj_CTMIP.nii.gz','Proj_Vessel.nii.gz')).get_fdata() 
                ct_topkmip_proj_list.append(ct_topkmip_proj)
                vessel_proj_list.append(vessel_proj)

                '# Record the inverse rotation matrix for each view in the sequence. (Inversely rotate the predicted nodes feature at view x to origin view 0.)'
                current_view = int(re.findall(r'\d+',sorted_paths[inner_seq_idx].split('/')[-1])[0])
                inverse_theta = inverse_rot_matrix(current_view=current_view, origin_view=0, views=180) # (3,4)
                forward_theta = forward_rot_matrix(current_view=current_view, origin_view=0, views=180) # (3,4)
                inverse_theta_list.append(inverse_theta)
                forward_theta_list.append(forward_theta)

            for inverse_theta, forward_theta in zip(inverse_theta_list,forward_theta_list):
                rot_mat['inverse_rot'].append(inverse_theta)
                rot_mat['forward_rot'].append(forward_theta)

            ct_topkmip_proj_list = np.concatenate(np.array(ct_topkmip_proj_list)[:,:,None,], -2).reshape(256,256,-1)
            vessel_proj_list = np.concatenate(np.array(vessel_proj_list), -1) # (256,256,10)

            os.makedirs(data_root.replace('ProjCTMIP_256_256_1','TrainSeq_ProjCTtopkMIP_'+str(sequence_num)+'views'), exist_ok=True)
            os.makedirs(data_root.replace('ProjCTMIP_256_256_1','TrainSeq_ProjVessel_'+str(sequence_num)+'views'), exist_ok=True)
            os.makedirs(data_root.replace('ProjCTMIP_256_256_1','TrainSeq_ProjRotMat_'+str(sequence_num)+'views'), exist_ok=True)

            save_nifti(ct_topkmip_proj_list, os.path.join(data_root.replace('ProjCTMIP_256_256_1','TrainSeq_ProjCTtopkMIP_'+str(sequence_num)),patient_idx+'_'+'Seq'+sequence_idx+'.nii.gz'))
            save_nifti(vessel_proj_list, os.path.join(data_root.replace('ProjCTMIP_256_256_1','TrainSeq_ProjVessel_'+str(sequence_num)+'views'),patient_idx+'_'+'Seq'+sequence_idx+'.nii.gz'))
            np.save(os.path.join(data_root.replace('ProjCTMIP_256_256_1','TrainSeq_ProjRotMat_'+str(sequence_num)+'views'),patient_idx+'_'+'Seq'+sequence_idx+'.npy'), rot_mat)

if __name__ == "__main__":
    CT_resize()
    projections()
    cat_projections_batch()