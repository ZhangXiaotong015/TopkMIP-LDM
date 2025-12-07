import os
import SimpleITK as sitk
import re
import numpy as np
import skimage.transform as st

def prediction_transpose(root_path=None):

    root_path = r'/model/latent_diffusion/outputs/Abla1_IRCADB_leave1out/Abla1_IRCADBExp_plain_leave1out_F0/recons'

    for root, dirs, files in os.walk(root_path): 
        for name in files:
            try:
                pred_vol = sitk.ReadImage(os.path.join(root,name))
            except:
                os.remove(os.path.join(root,name))
                continue
            origin = pred_vol.GetOrigin()
            space = pred_vol.GetSpacing()
            direction = pred_vol.GetDirection()
            try:
                pred_vol = sitk.GetArrayFromImage(pred_vol).transpose(2,1,0)# # (256,256,D)
            except:
                os.remove(os.path.join(root,name))
                continue
        
            save_path = os.path.join(root,name)
            img = sitk.GetImageFromArray(pred_vol.transpose(2,0,1))
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(space)
            sitk.WriteImage(img, '{}'.format(save_path))

def resolution_recovery(predict_root=None):
    ori_root = r'/3DircadbNifti_newLiverMask/patientCTVolume'
    predict_root = r'/model/latent_diffusion/outputs_ISBI/Abla1_IRCADB_leave1out/Abla1_IRCADBExp_plain_leave1out_F0/recons_singleTermOpt'

    resized_img_shape = 256
    for root, dirs, files in os.walk(predict_root): 
        for name in files:
            if 'resolution_recovery' in root.split('/')[-1]:
                continue
            if 'noiseCancelConnect' not in name:
                continue

            pred_vol = sitk.ReadImage(os.path.join(root,name))
            pred_vol = sitk.GetArrayFromImage(pred_vol).transpose(1,2,0)# # (256,256,D)

            ct_ori = sitk.ReadImage(os.path.join(ori_root, 'PATIENT'+str(int(re.findall(r"\d+",name)[0])).zfill(2)+'_DICOM.nii.gz'))
            origin = ct_ori.GetOrigin()
            space = ct_ori.GetSpacing()
            direction = ct_ori.GetDirection()
            ct_ori = sitk.GetArrayFromImage(ct_ori).transpose(1,2,0) # (H,W,D)

            liver_ori = sitk.ReadImage(os.path.join(ori_root.replace('patientCTVolume','liver'), 'liver'+re.findall(r"\d+",str(int(re.findall(r"\d+",name)[0])).zfill(2))[0]+'.nii.gz'))
            liver_ori = sitk.GetArrayFromImage(liver_ori).transpose(1,2,0) # (H,W,D)
            liver_ori[liver_ori>0] = 1

            liver_dist = liver_ori
            liver_dist[liver_dist>0] = 1
            edge = np.argwhere(liver_dist==1)
            row_min = edge[:,0].min()
            row_max = edge[:,0].max()
            col_min = edge[:,1].min()
            col_max = edge[:,1].max()
            dep_min = edge[:,2].min()
            dep_max = edge[:,2].max()

            cropped_ct = ct_ori[row_min:row_max, col_min:col_max, dep_min:dep_max]
            # cropped_ct = ct_ori[col_min:col_max, row_min:row_max, :]

            H,W,D = cropped_ct.shape

            pred_ori_liverRegion = st.resize(pred_vol, (H,W,D), order=0, preserve_range=True, anti_aliasing=False)

            pred_ori = np.zeros((ct_ori.shape[0], ct_ori.shape[1], ct_ori.shape[2]))
            pred_ori[row_min:row_max, col_min:col_max, dep_min:dep_max] = pred_ori_liverRegion
            # pred_ori[col_min:col_max, row_min:row_max, :] = pred_ori_liverRegion

            pred_ori[pred_ori<0] = 0
            pred_ori = (pred_ori-pred_ori.min()) / (pred_ori.max()-pred_ori.min())

            os.makedirs(root.replace('validation_','resolution_recovery_validation_'), exist_ok=True)
            save_path = os.path.join(root.replace('validation_','resolution_recovery_validation_'),name)
            img = sitk.GetImageFromArray(pred_ori.transpose(2,0,1)*liver_ori.transpose(2,0,1))
            # img = sitk.GetImageFromArray(pred_ori.transpose(2,1,0))
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(space)
            sitk.WriteImage(img, '{}'.format(save_path))

if __name__ == "__main__":
    prediction_transpose()
    resolution_recovery()