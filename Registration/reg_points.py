from utils import *
import pandas as pd
def read_reg_mat(file):
    with open(file, 'r') as f:
        l = [[float(num) for num in line.replace(' \n', '').split(' ')] for line in f]
    return np.array(l)

# df = pd.read_csv('../Processing/data_list_in_analysis_osc_curated.csv')
df = pd.read_csv('../Datalists/data_list_in_analysis_slimmed_v4.csv')

atlas_path = r'/groups/ahrens/ahrenslab/jing/zebrafish_atlas/yumu_confocal/20150519/im/cy14_1p_stitched.h5'
atlas = np.swapaxes(read_h5(atlas_path, dset_name='channel0'),1,2).astype('float64')[::-1]
fatlas_smooth = atlas.copy()
thresh_u = np.percentile(fatlas_smooth, 99)
thresh_l = np.percentile(fatlas_smooth, 10)
fatlas_smooth[fatlas_smooth>thresh_u] = thresh_u
fatlas_smooth = fatlas_smooth - thresh_l
fatlas_smooth[fatlas_smooth<0] = 0
fatlas_smooth = fatlas_smooth/fatlas_smooth.max()
atlas = None

for ind, row in df.iterrows():
    if row['high_res']=='None':
        continue
    if ind<29:
        continue
    if ind>31:
        continue
    print(ind)
    moving_root = row['dat_dir']
    fimg_dir = row['save_dir']
    save_root = fimg_dir+'/registration/'
    write_path = row['high_res']
    fimg = np.load(fimg_dir + 'Y_ave.npy').squeeze().astype('float')
    fix = read_h5(write_path).astype('float')
    if os.path.exists(fimg_dir + 'cell_center_registered.npy'):
        continue
    
    fimg_smooth = fimg.copy()
    thresh_u = np.percentile(fimg, 99)
    thresh_l = np.percentile(fimg, 10)
    fimg_smooth[fimg_smooth>thresh_u] = thresh_u
    fimg_smooth = fimg_smooth - thresh_l
    fimg_smooth[fimg_smooth<0] = 0
    fimg_smooth = fimg_smooth/fimg_smooth.max()
    fimg = None

    ffix_smooth = fix.copy()
    thresh_u = np.percentile(ffix_smooth, 99)
    thresh_l = np.percentile(ffix_smooth, 10)
    ffix_smooth[ffix_smooth>thresh_u] = thresh_u
    ffix_smooth = ffix_smooth - thresh_l
    ffix_smooth[ffix_smooth<0] = 0
    ffix_smooth = ffix_smooth/ffix_smooth.max()
    fix = None
    
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
    x_, y_, z_ = fix_range
    ffix_smooth_ = ffix_smooth[z_[0]:z_[1], y_[0]:y_[1], x_[0]:x_[1]]
    ffix_smooth_ = ffix_smooth_[:, :, ::(1-flip_x*2)]
    fimg_smooth  = fimg_smooth[:, y_[0]:y_[1], x_[0]:x_[1]]
    x_, y_, z_ = atlas_range
    fatlas_smooth_ = fatlas_smooth[z_[0]:z_[1], y_[0]:y_[1], x_[0]:x_[1]]
    saltas_ = sitk.GetImageFromArray(fatlas_smooth_)
    sfix_ = sitk.GetImageFromArray(ffix_smooth_)
    sfimg_ = sitk.GetImageFromArray(fimg_smooth[:, :, ::(1-flip_x*2)])
    fatlas_smooth_, ffix_smooth = None, None
    ffix_smooth_ = None
    fimg_smooth = None

    sfimg_.SetSpacing(fimg_vox[:-1])
    sfix_.SetSpacing(fix_vox[:-1])
    saltas_.SetSpacing(atlas_vox[:-1])

    sfimg_.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    sfix_.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    saltas_.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    output = fimg_dir + 'registration/atlas_fix_wrap_mat.nii.gz'
    output_inv = fimg_dir + 'registration/atlas_fix_wrap_inv_mat.nii.gz'
    atlas_fix_wrap_mat = nib.load(output)
    atlas_fix_wrap_inv_mat = nib.load(output_inv)
    affine_output = fimg_dir + 'registration/atlas_fix_affine.mat'
    rigid_output = fimg_dir + 'registration/atlas_fix_rigid.mat'
    hl_output = fimg_dir + 'registration/fimg_fix_affine.mat'
    affine_mat = read_reg_mat(affine_output)
    rigid_mat = read_reg_mat(rigid_output)
    hl_affine_mat = read_reg_mat(hl_output)

    displacementField=nib.load(output)
    displacement_image = sitk.GetImageFromArray(displacementField.get_data().squeeze().astype('float'), isVector=True)
    tx_wrap = sitk.DisplacementFieldTransform(displacement_image)
    displacementField = None

    displacementField = nib.load(output_inv)
    displacement_image = sitk.GetImageFromArray(displacementField.get_data().squeeze().astype('float'), isVector=True)
    tx_wrap_inv = sitk.DisplacementFieldTransform(displacement_image)
    displacementField = None

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

    _ = sitk.Resample(sfimg_, sfix_, tx_res)
    _ = sitk.Resample(_, saltas_, tx_affine_inv)
    satlas_wrap = sitk.Resample(_, saltas_, tx_wrap_inv)
    _ = None
        
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
    # z, x, y = cell_centers.T
    x_, y_, z_ = fix_range
    if y_[0]>0:
        y = y-y_[0]
        z = z[y>0]
        x = x[y>0]
        y = y[y>0]
    if flip_x == 1:
        # flip x when it is marked
        x_, y_, z_ = sfimg_.GetSize()
        x = x_ - x
    point_list_low = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T.astype('float')
    paltas_low = [tx_inv_high_atlas(tx_inv_low_high(_)) for _ in point_list_low]
    paltas_low = np.array(paltas_low)
    x, y, z = paltas_low.T
    cell_centers_ = np.vstack([z.flatten(), y.flatten(), x.flatten()]).T.astype('float')
    np.save(fimg_dir + 'cell_center_registered.npy', cell_centers_)

    