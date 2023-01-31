from utils import *
import pandas as pd
df = pd.read_csv('../Datalists/data_list_in_analysis_glia_v3.csv')

atlas_path = r'/groups/ahrens/ahrenslab/jing/zebrafish_atlas/yumu_confocal/20150519/im/cy14_1p_stitched.h5'
atlas = np.swapaxes(read_h5(atlas_path, dset_name='channel0'),1,2).astype('float64')[::-1]
atlas = atlas.astype('int16')
fatlas_smooth = atlas.copy()
thresh_u = np.percentile(fatlas_smooth, 99)
thresh_l = np.percentile(fatlas_smooth, 10)
fatlas_smooth[fatlas_smooth>thresh_u] = thresh_u
fatlas_smooth = fatlas_smooth - thresh_l
fatlas_smooth[fatlas_smooth<0] = 0
fatlas_smooth = fatlas_smooth/fatlas_smooth.max()
atlas = None


ind = 9
row = df.iloc[ind]
moving_root = row['dat_dir']
fimg_dir = row['save_dir']
save_root = fimg_dir+'/registration/'
write_path = row['high_res']

row_neuron = df.iloc[ind-1]
write_path_neuron = row_neuron['high_res']
fimg_dir_neuron = row_neuron['save_dir']

fimg = np.load(fimg_dir + 'Y_ave.npy').squeeze().astype('float')
if fimg.max()<100:
    fimg[fimg<0] = 0
    fimg= (fimg*1000).astype('int16')
fix = read_h5(write_path).astype('int16')

fimg_smooth = fimg.copy()
thresh_u = np.percentile(fimg, 99)
thresh_l = np.percentile(fimg, 40)
fimg_smooth[fimg_smooth>thresh_u] = thresh_u
fimg_smooth = fimg_smooth - thresh_l
fimg_smooth[fimg_smooth<0] = 0
fimg_smooth = fimg_smooth/fimg_smooth.max()
fimg = None

ffix_smooth = fix.copy()
thresh_u = np.percentile(ffix_smooth, 99)
thresh_l = np.percentile(ffix_smooth, 40)
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


fix_neuron = read_h5(write_path_neuron).astype('int16')
ffix_smooth_neuron = fix_neuron.copy()
thresh_u = np.percentile(ffix_smooth_neuron, 99)
thresh_l = np.percentile(ffix_smooth_neuron, 40)
ffix_smooth_neuron[ffix_smooth_neuron>thresh_u] = thresh_u
ffix_smooth_neuron = ffix_smooth_neuron - thresh_l
ffix_smooth_neuron[ffix_smooth_neuron<0] = 0
ffix_smooth_neuron = ffix_smooth_neuron/ffix_smooth_neuron.max()
fix_neuron = None

_ = np.load(fimg_dir_neuron+'registration/sample_parameters.npz', allow_pickle=True)
fix_neuron_range = _['fix_range']
fix_neuron_vox = _['fix_vox']
try:
    flip_xyz = _['flip_xyz']
except:
    flip_xyz = np.array([0, 0, 0]).astype('bool')
flip_x = flip_xyz[0].astype('int')
x_, y_, z_ = fix_neuron_range
ffix_neuron_smooth_ = ffix_smooth_neuron[z_[0]:z_[1], y_[0]:y_[1], x_[0]:x_[1]]
ffix_neuron_smooth_ = ffix_neuron_smooth_[:, :, ::(1-flip_x*2)]
sfix_neuron = sitk.GetImageFromArray(ffix_neuron_smooth_)
ffix_neuron_smooth_ = None
sfix_neuron.SetSpacing(fix_neuron_vox[:-1])
sfix_neuron.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])


neuron_affine_output = fimg_dir + 'registration/neuron_atlas_fix_affine.mat'
neuron_rigid_output = fimg_dir + 'registration/neuron_atlas_fix_rigid.mat'
neuron_fix_output = fimg_dir + 'registration/neuron_fix_rigid.mat'
hl_output = fimg_dir + 'registration/fimg_fix_affine.mat'

neuron_affine_mat = read_reg_mat(neuron_affine_output)
neuron_rigid_mat = read_reg_mat(neuron_rigid_output)
neuron_fix_mat = read_reg_mat(neuron_fix_output)
hl_affine_mat = read_reg_mat(hl_output)

output = fimg_dir + 'registration/neuron_atlas_fix_wrap_mat.nii.gz'
output_inv = fimg_dir + 'registration/neuron_atlas_fix_wrap_inv_mat.nii.gz'

displacementField=nib.load(output)
displacement_image = sitk.GetImageFromArray(displacementField.get_data().squeeze().astype('float'), isVector=True)
tx_wrap = sitk.DisplacementFieldTransform(displacement_image)
displacementField = None
displacementField = nib.load(output_inv)
displacement_image = sitk.GetImageFromArray(displacementField.get_data().squeeze().astype('float'), isVector=True)
tx_wrap_inv = sitk.DisplacementFieldTransform(displacement_image)
displacementField = None

dims = 3
tx_neuron_affine = sitk.AffineTransform(dims)
tx_neuron_affine.SetMatrix(neuron_affine_mat[:3, :3].reshape(-1))
tx_neuron_affine.SetTranslation(neuron_affine_mat[:3, 3])
tx_neuron_affine_inv = tx_neuron_affine.GetInverse()

dims = 3
tx_neuron_rigid = sitk.AffineTransform(dims)
tx_neuron_rigid.SetMatrix(neuron_rigid_mat[:3, :3].reshape(-1))
tx_neuron_rigid.SetTranslation(neuron_rigid_mat[:3, 3])
tx_neuron_rigid_inv = tx_neuron_rigid.GetInverse()


dims = 3
tx_neuron_fix = sitk.AffineTransform(dims)
tx_neuron_fix.SetMatrix(neuron_fix_mat[:3, :3].reshape(-1))
tx_neuron_fix.SetTranslation(neuron_fix_mat[:3, 3])
tx_neuron_fix_inv = tx_neuron_fix.GetInverse()


dims = 3
tx_res = sitk.AffineTransform(dims)
tx_res.SetMatrix(hl_affine_mat[:3, :3].reshape(-1))
tx_res.SetTranslation(hl_affine_mat[:3, 3])
tx_res_inv = tx_res.GetInverse()


def tx_inv_high_atlas(point_):
    _ = sfix_.TransformContinuousIndexToPhysicalPoint(point_)
    _ = tx_neuron_fix.TransformPoint(_)
    _ = tx_neuron_affine.TransformPoint(_)
    _ = tx_wrap.TransformPoint(_)
    return saltas_.TransformPhysicalPointToContinuousIndex(_)

def tx_inv_low_high(point_):
    _ = sfimg_.TransformContinuousIndexToPhysicalPoint(point_)
    _ = tx_res_inv.TransformPoint(_)
    return sfix_.TransformPhysicalPointToContinuousIndex(_)

cell_centers = np.load(fimg_dir + 'cell_center.npy')
# z, y, x = cell_centers.T
z, x, y = cell_centers.T
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


