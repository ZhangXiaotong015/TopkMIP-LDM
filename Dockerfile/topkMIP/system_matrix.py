import pickle
import numpy as np
import math
import scipy.sparse as sp
import os

def system_matrix_creation(save_root=None,sysmat_num_splits=30):
    def forward_rot_matrix(current_view, origin_view=0, views=180):
        unit_interval = math.pi/views
        angle = current_view * unit_interval - origin_view * unit_interval   
        # axis='z'
        theta = np.array([[np.cos(angle), -np.sin(angle), np.zeros(angle.shape)],
                          [np.sin(angle), np.cos(angle), np.zeros(angle.shape)],
                          [np.zeros(angle.shape), np.zeros(angle.shape), np.ones(angle.shape)]])   
        return theta 

    def forward_projection_matrix(rotation_matrices, vol_size=256, det_size=256):
        """
        Compute the sparse system matrix for CT reconstruction using parallel beam geometry.

        Args:
            rotation_matrices (numpy.ndarray): Array of shape (180, 3, 3) representing rotation matrices for each view.
            vol_size (int): Size of the 3D volume (default is 256).
            det_size (int): Size of the 2D detector (default is 256).

        Returns:
            scipy.sparse.csr_matrix: Sparse system matrix of shape (num_views * det_size^2, vol_size^3).
        """
        num_views = rotation_matrices.shape[0]
        
        # Create a grid of voxel coordinates
        x = np.arange(vol_size) - vol_size / 2
        y = np.arange(vol_size) - vol_size / 2
        z = np.arange(vol_size) - vol_size / 2
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Flatten the grid into a list of voxel coordinates
        voxel_coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

        # Initialize lists to store the data, row indices, and column indices for the sparse matrix
        data = []
        row_indices = []
        col_indices = []
        
        # Loop through each view
        for view in range(num_views):
            rotation_matrix = rotation_matrices[view]
            
            # Rotate the voxel coordinates
            rotated_coords = np.dot(rotation_matrix, voxel_coords)
            
            # Project the rotated voxel coordinates onto the xz plane
            det_x = np.floor(rotated_coords[0, :] + det_size / 2).astype(int)
            det_z = np.floor(rotated_coords[2, :] + det_size / 2).astype(int)
            
            # Filter out the out-of-bounds coordinates
            in_bounds = np.logical_and.reduce((det_x >= 0, det_x < det_size, det_z >= 0, det_z < det_size))
            
            # Compute linear indices for the valid detector and voxel coordinates
            valid_detector_indices = view * det_size * det_size + det_z[in_bounds] * det_size + det_x[in_bounds]
            valid_voxel_indices = np.arange(vol_size**3)[in_bounds]
            
            # Store the valid entries in the sparse matrix format
            data.extend(np.ones(len(valid_detector_indices)))
            row_indices.extend(valid_detector_indices)
            col_indices.extend(valid_voxel_indices)
        
        # Create the sparse system matrix in CSR format
        system_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_views * det_size * det_size, vol_size * vol_size * vol_size))
        
        return system_matrix

    def back_projection_matrices(system_matrices):
        """
        Compute the back-projection matrices (transposes of the system matrices).

        Args:
            system_matrices (list of scipy.sparse.csr_matrix): List of sparse matrices for forward projection.

        Returns:
            list of scipy.sparse.csr_matrix: List of sparse matrices for back-projection.
        """
        # Create a list to hold the back-projection matrices
        back_proj_matrices = []
        
        # Loop through each system matrix and transpose it
        for matrix in system_matrices:
            back_proj_matrices.append(matrix.transpose())
        
        return back_proj_matrices

    'Calculate and save forward projection matrix'
    splits = 180
    system_matrices = []
    for idx in range(splits):
        print(f'Forward projection matrix: {idx+1}/{splits}')
        current_view = np.arange(int((180/splits)*idx), int((180/splits)*(idx+1)))
        rotation_matrices = forward_rot_matrix(current_view).transpose(2,0,1)
        system_matrix_sub = forward_projection_matrix(rotation_matrices, vol_size=256, det_size=256)
        # p = system_matrix_sub.dot(u.ravel()) 
        # p = p.reshape(256,256)
        # if idx%30==0:
        #     plt.imshow(p,'gray')
        #     plt.savefig(r'/tempfiles/proj_'+str(idx)+'.png')
        system_matrices.append(system_matrix_sub) 
    # # Save the list to a file
    # with open(r'/system_matrix/sparse_matrices_forward.pkl', 'wb') as f:
    #     pickle.dump(system_matrices, f)

    'Calculate and save back projection matrix'
    system_matrices_back = back_projection_matrices(system_matrices)
    # # Save the list to a file
    # with open(r'/system_matrix/sparse_matrices_back.pkl', 'wb') as f:
    #     pickle.dump(system_matrices_back, f)

    # '# Load forward projection matrix'
    # with open(r'/system_matrix/sparse_matrices_forward.pkl', 'rb') as f:
    #     system_matrices = pickle.load(f)
    # '# Load back projection matrix'
    # with open(r'/system_matrix/sparse_matrices_back.pkl', 'rb') as f:
    #     system_matrices_back = pickle.load(f)

    '# Split into 30 submatrices along the first axis'
    # num_splits = 30
    split_matrices_forward = np.array_split(system_matrices, sysmat_num_splits)
    split_matrices_back = np.array_split(system_matrices_back, sysmat_num_splits)
    # Save each submatrix as a separate .pkl file
    for idx, submatrix in enumerate(split_matrices_forward):
        with open(os.path.join(save_root,f'sub_sparse_matrices_forward_{idx}.pkl'), 'wb') as f:
            pickle.dump(submatrix, f)
    for idx, submatrix in enumerate(split_matrices_back):
        with open(os.path.join(save_root,f'sub_sparse_matrices_back_{idx}.pkl'), 'wb') as f:
            pickle.dump(submatrix, f)

if __name__ == "__main__":

    system_matrix_creation() # Perform only once