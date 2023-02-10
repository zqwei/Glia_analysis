from utils import *
from skimage.io import imread
import pandas as pd

atlas_path = r'/groups/ahrens/ahrenslab/jing/zebrafish_atlas/yumu_confocal/20150519/im/cy14_1p_stitched.h5'
atlas = np.swapaxes(read_h5(atlas_path, dset_name='channel0'),1,2).astype('float64')[::-1]

zatlas = imread('/nrs/ahrens/Ziqiang/Atlas/Single_cell/T_AVG_jf5Tg.tif')
zatlas = np.swapaxes(zatlas, 1, 2)

fatlas_smooth = atlas.copy()
thresh_u = np.percentile(fatlas_smooth, 99)
thresh_l = np.percentile(fatlas_smooth, 10)
fatlas_smooth[fatlas_smooth>thresh_u] = thresh_u
fatlas_smooth = fatlas_smooth - thresh_l
fatlas_smooth[fatlas_smooth<0] = 0
fatlas_smooth = fatlas_smooth/fatlas_smooth.max()


ffix_smooth = zatlas.copy()
thresh_u = np.percentile(ffix_smooth, 99)
thresh_l = np.percentile(ffix_smooth, 10)
ffix_smooth[ffix_smooth>thresh_u] = thresh_u
ffix_smooth = ffix_smooth - thresh_l
ffix_smooth[ffix_smooth<0] = 0
ffix_smooth = ffix_smooth/ffix_smooth.max()
fix = None

atlas_zvox = 2.
spim_res = 0.406
fix_vox = np.array([1.0, 1.0, 1.0, 1])
atlas_vox = np.array([spim_res, spim_res, atlas_zvox, 1])

save_tmp = '/nrs/ahrens/Ziqiang/scratch/registration/'
save_root = '/nrs/ahrens/Ziqiang/Atlas/Single_cell/T_AVG_jf5Tg_inv/'

if not os.path.exists(save_root):
    os.mkdir(save_root)
    
print(save_root)

ffix_smooth_ = ffix_smooth[:,:,:]
fatlas_smooth_ = fatlas_smooth[:170, :, :]
altas_ = nib.Nifti1Image(fatlas_smooth_.swapaxes(0, 2), affine=np.diag(atlas_vox))
fix_ = nib.Nifti1Image(ffix_smooth_.swapaxes(0, 2), affine=np.diag(fix_vox))
nib.save(altas_, save_tmp+'altas.nii.gz')
nib.save(fix_, save_tmp+'fix.nii.gz')

fz, fy, fx = ffix_smooth_.shape
az, ay, ax = fatlas_smooth_.shape

np.savez(save_root+'sample_parameters', fix_range=np.array([[ 0, fx],[0, fy], [0, fz]]), \
                                        atlas_range=np.array([[0, ax], [ 0, ay], [0, 170]]), \
                                        flip_xyz = np.array([0, 0, 0]).astype('bool'))

altas_file = save_tmp+'fix.nii.gz'
fix_file = save_tmp+'altas.nii.gz'

altas_ = nib.load(altas_file)
fix_ = nib.load(fix_file)

## rigid
output = save_root + 'atlas_fix_rigid.mat'
greedy = 'greedy -d 3 -dof 6 -a -m MI -i ' + fix_file + ' ' + altas_file + ' -o ' + output + ' -ia-image-centers -n 200x100x50'
os.system(greedy)

## Check results from rigid
output = save_root + 'atlas_fix_rigid.mat'
atlas_fix_rigid_align = save_tmp+'atlas_fix_rigid.nii.gz'
greedy_reg = 'greedy -d 3 -rf ' + fix_file + ' -rm ' + altas_file + ' ' + atlas_fix_rigid_align + ' -ri LINEAR -r ' + output
os.system(greedy_reg)

atlas_rigid_reg = nib.load(save_tmp+'atlas_fix_rigid.nii.gz')
plt.imshow(fix_.get_data().max(2), cmap='Greens')
plt.imshow(atlas_rigid_reg.get_data().max(2), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.imshow(fix_.get_data().max(0), cmap='Greens')
plt.imshow(atlas_rigid_reg.get_data().max(0), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.imshow(fix_.get_data().max(1), cmap='Greens')
plt.imshow(atlas_rigid_reg.get_data().max(1), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()



## affine
rigid_output = save_root + 'atlas_fix_rigid.mat'
output = save_root + 'atlas_fix_affine.mat'
greedy = 'greedy -d 3 -dof 12 -a -m NMI -i ' + fix_file + ' ' + altas_file + ' -o ' + output + ' -ia ' + rigid_output + ' -n 50x20x10'
os.system(greedy)

## Check results from affine
atlas_fix_affine_align = save_tmp+'atlas_fix_affine.nii.gz'
output = save_root + 'atlas_fix_affine.mat'
greedy_reg = 'greedy -d 3 -rf ' + fix_file + ' -rm ' + altas_file + ' ' + atlas_fix_affine_align + ' -ri LINEAR -r ' + output
os.system(greedy_reg)

atlas_affine_reg = nib.load(save_tmp+'atlas_fix_affine.nii.gz')
plt.imshow(fix_.get_data().max(2), cmap='Greens')
plt.imshow(atlas_affine_reg.get_data().max(2), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.imshow(fix_.get_data().max(0), cmap='Greens')
plt.imshow(atlas_affine_reg.get_data().max(0), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.imshow(fix_.get_data().max(1), cmap='Greens')
plt.imshow(atlas_affine_reg.get_data().max(1), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()




## deformable
affine_output = save_root + 'atlas_fix_affine.mat'
output = save_root + 'atlas_fix_wrap_mat.nii.gz'
output_inv = save_root + 'atlas_fix_wrap_inv_mat.nii.gz'
smoothing = '1.7vox 0.7vox' # Deformable registration smoothing parameters
greedy = 'greedy -d 3 -m WNCC 4x4x4 -i ' + fix_file + ' ' + altas_file + ' -it ' + affine_output + ' -o ' + output + ' -n 300x40x20 -s ' + smoothing + ' -oinv ' + output_inv
os.system(greedy)

## Check results from deformable
atlas_fix_wrap_align = save_tmp+'atlas_fix_wrap.nii.gz'
greedy = 'greedy -d 3 -rf ' + fix_file + ' -rm ' + altas_file + ' ' + atlas_fix_wrap_align + ' -ri LINEAR -r ' + output + ' ' + affine_output
os.system(greedy)
# #### invert
atlas_fix_affine_wrap_inv = save_tmp+'atlas_fix_wrap_inv.nii.gz'
greedy = 'greedy -d 3 -rf ' + altas_file + ' -rm ' + fix_file + ' ' + atlas_fix_affine_wrap_inv + ' -r ' + affine_output + ',-1 ' + output_inv
os.system(greedy)


atlas_wrap_reg = nib.load(save_tmp+'atlas_fix_wrap.nii.gz')
plt.figure(figsize=(20, 10))
plt.imshow(.get_data().max(2), cmap='Greens')
plt.imshow(atlas_wrap_reg.get_data().max(2), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.figure(figsize=(20, 10))
plt.imshow(fix_.get_data().max(0), cmap='Greens')
plt.imshow(atlas_wrap_reg.get_data().max(0), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

plt.figure(figsize=(20, 10))
plt.imshow(fix_.get_data().max(1), cmap='Greens')
plt.imshow(atlas_wrap_reg.get_data().max(1), cmap='Reds', alpha=0.5)
plt.axis('off')
plt.show()

## remove temporal results
clean_folder = 'rm -rf /nrs/ahrens/Ziqiang/scratch/registration/*.*'
os.system(clean_folder)