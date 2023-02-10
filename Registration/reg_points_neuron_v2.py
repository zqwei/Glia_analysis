from utils import *
import pandas as pd

# df = pd.read_csv('../Datalists/data_list_in_analysis_slimmed_v4.csv')
# df = pd.read_csv('../Datalists/data_list_in_analysis_glia_v3.csv')
df = pd.read_csv('../Datalists/data_list_in_analysis_NE_v2.csv')
# df = pd.read_csv('../Datalists/data_list_in_analysis_DA_v0.csv')


for ind, row in df.iterrows():
    if ind not in [4, 6, 8]:
        continue
    if row['high_res']=='None':
        continue
    fimg_dir = row['save_dir']
    save_root = fimg_dir+'/registration/'
    if not os.path.exists(fimg_dir+'cell_center_voluseg.npy'):
        # redo files with x, y swap
        continue
    # if os.path.exists(fimg_dir+'cell_center_registered_.npy'):
    #     continue
    if not os.path.exists(save_root):
        continue
    if not os.path.exists(fimg_dir + 'registration/atlas_fix_wrap_mat.nii.gz'):
        continue
    print(ind)
    
    fimg = np.load(fimg_dir + 'Y_ave.npy').squeeze()
    _ = np.load(fimg_dir+'registration/sample_parameters.npz', allow_pickle=True)
    fix_range = _['fix_range']
    atlas_range = _['atlas_range']
    fimg_vox = _['fimg_vox']
    fix_vox = _['fix_vox']
    atlas_vox = _['atlas_vox']
    try:
        flip_xyz = _['flip_xyz']
    except:
        flip_xyz = np.array([0, 0, 0]).astype('bool')
    flip_x = flip_xyz[0].astype('int')
    # create empty image for size information
    x_, y_, z_ = atlas_range
    # item() to convert numpy.int to python int
    # sitk GetImageFromArray swap z & x
    saltas_ = sitk.Image((x_[1]-x_[0]).item(), (y_[1]-y_[0]).item(), \
                         (z_[1]-z_[0]).item(), sitk.sitkUInt8)  
    x_, y_, z_ = fix_range
    sfix_ = sitk.Image((x_[1]-x_[0]).item(), (y_[1]-y_[0]).item(), \
                       (z_[1]-z_[0]).item(), sitk.sitkUInt8)
    sfimg_ = sitk.Image(fimg.shape[2], fimg.shape[1], \
                        fimg.shape[0], sitk.sitkUInt8)

    sfimg_.SetSpacing(fimg_vox[:-1])
    sfix_.SetSpacing(fix_vox[:-1])
    saltas_.SetSpacing(atlas_vox[:-1])

    sfimg_.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    sfix_.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    saltas_.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    # transforms (affines)
    affine_output = fimg_dir + 'registration/atlas_fix_affine.mat'
    rigid_output = fimg_dir + 'registration/atlas_fix_rigid.mat'
    hl_output = fimg_dir + 'registration/fimg_fix_affine.mat'
    affine_mat = read_reg_mat(affine_output)
    rigid_mat = read_reg_mat(rigid_output)
    hl_affine_mat = read_reg_mat(hl_output)
    
    dims = 3
    tx_affine = sitk.AffineTransform(dims)
    tx_affine.SetMatrix(affine_mat[:3, :3].reshape(-1))
    tx_affine.SetTranslation(affine_mat[:3, 3])
    tx_affine_inv = tx_affine.GetInverse()

    dims = 3
    tx_rigid = sitk.AffineTransform(dims)
    tx_rigid.SetMatrix(rigid_mat[:3, :3].reshape(-1))
    tx_rigid.SetTranslation(rigid_mat[:3, 3])
    tx_rigid_inv = tx_rigid.GetInverse()

    dims = 3
    tx_res = sitk.AffineTransform(dims)
    tx_res.SetMatrix(hl_affine_mat[:3, :3].reshape(-1))
    tx_res.SetTranslation(hl_affine_mat[:3, 3])
    tx_res_inv = tx_res.GetInverse()
    
    # transforms (warp)
    output = fimg_dir + 'registration/atlas_fix_wrap_mat.nii.gz'
    output_inv = fimg_dir + 'registration/atlas_fix_wrap_inv_mat.nii.gz'
    displacementField=nib.load(output)
    displacement_image = sitk.GetImageFromArray(displacementField.get_data().squeeze().astype('float'), isVector=True)
    tx_wrap = sitk.DisplacementFieldTransform(displacement_image)
    displacementField = None

    # displacementField = nib.load(output_inv)
    # displacement_image = sitk.GetImageFromArray(displacementField.get_data().squeeze().astype('float'), isVector=True)
    # tx_wrap_inv = sitk.DisplacementFieldTransform(displacement_image)
    # displacementField = None
        
    def tx_inv_high_atlas(point_):
        _ = sfix_.TransformContinuousIndexToPhysicalPoint(point_)
        _ = tx_affine.TransformPoint(_)
        _ = tx_wrap.TransformPoint(_)
        return saltas_.TransformPhysicalPointToContinuousIndex(_)

    def tx_inv_low_high(point_):
        _ = sfimg_.TransformContinuousIndexToPhysicalPoint(point_)
        _ = tx_res_inv.TransformPoint(_)
        return sfix_.TransformPhysicalPointToContinuousIndex(_)

    cell_centers = np.load(fimg_dir + 'cell_center.npy')
    z, y, x = cell_centers.T
    x_, y_, z_ = fix_range
    if y_[0]>0:
        y = y-y_[0]
        z = z[y>0]
        x = x[y>0]
        y = y[y>0]
    if flip_x == 1:
        # flip x when it is marked
        x_, y_, z_ = sfimg_.GetSize()
        print(x_, y_, z_)
        x = x_ - x
    point_list_low = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T.astype('float')
    paltas_low = [tx_inv_high_atlas(tx_inv_low_high(_)) for _ in point_list_low]
    paltas_low = np.array(paltas_low)
    x, y, z = paltas_low.T
    cell_centers_ = np.vstack([z.flatten(), y.flatten(), x.flatten()]).T.astype('float')
    if os.path.exists(fimg_dir + 'cell_center_registered.npy'):
        os.rename(fimg_dir + 'cell_center_registered.npy', fimg_dir + 'cell_center_registered_.npy')
    np.save(fimg_dir + 'cell_center_registered.npy', cell_centers_)

    