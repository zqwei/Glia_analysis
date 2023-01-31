from utils import *
import shutil
import pandas as pd
df = pd.read_csv('../Datalists/data_list_in_analysis_glia_v3.csv')

ind = 1
row = df.iloc[ind]
row_neuron = df.iloc[ind-1]

moving_root = row['dat_dir']
fimg_dir = row['save_dir']
fish_name = fimg_dir.split('/')[-3]+'_im_CM1'
if 'im_CM0_voluseg' in fimg_dir:
    fish_name = fimg_dir.split('/')[-3]+'_im_CM0_voluseg'
if 'im_CM1_voluseg' in fimg_dir:
    fish_name = fimg_dir.split('/')[-3]+'_im_CM1_voluseg'
if 'im_CM0_dff' in fimg_dir:
    fish_name = fimg_dir.split('/')[-3]+'_im_CM0_dff'
if 'im_CM1_dff' in fimg_dir:
    fish_name = fimg_dir.split('/')[-3]+'_im_CM1_dff'
if 'im_CM1_removal_osc' in fimg_dir:
    fish_name = fimg_dir.split('/')[-3]+'_im_CM1'
if 'im_CM0_removal_osc' in fimg_dir:
    fish_name = fimg_dir.split('/')[-3]+'_im_CM0'

save_root = fimg_dir+'/registration/'
pre_fix_root = '/nrs/ahrens/Ziqiang/scratch/registration/'+fish_name + '/'
print(save_root, pre_fix_root)
altas_file = pre_fix_root+'altas.nii.gz'
fix_file = pre_fix_root+'fix.nii.gz'
fimg_file = pre_fix_root+'fimg.nii.gz'

fimg_dir = row_neuron['save_dir']
fish_name = fimg_dir.split('/')[-3]+'_im_CM1'
if 'im_CM0_voluseg' in fimg_dir:
    fish_name = fimg_dir.split('/')[-3]+'_im_CM0_voluseg'
if 'im_CM1_voluseg' in fimg_dir:
    fish_name = fimg_dir.split('/')[-3]+'_im_CM1_voluseg'
if 'im_CM0_dff' in fimg_dir:
    fish_name = fimg_dir.split('/')[-3]+'_im_CM0_dff'
if 'im_CM1_dff' in fimg_dir:
    fish_name = fimg_dir.split('/')[-3]+'_im_CM1_dff'
if 'im_CM1_removal_osc' in fimg_dir:
    fish_name = fimg_dir.split('/')[-3]+'_im_CM1'
if 'im_CM0_removal_osc' in fimg_dir:
    fish_name = fimg_dir.split('/')[-3]+'_im_CM0'
    
save_root_neuron = fimg_dir+'/registration/'
pre_fix_root_neuron = '/nrs/ahrens/Ziqiang/scratch/registration/'+fish_name + '/'
fix_file_neuron = pre_fix_root_neuron+'fix.nii.gz'

altas_ = nib.load(pre_fix_root+'altas.nii.gz')
fix_ = nib.load(pre_fix_root+'fix.nii.gz')
fimg_ = nib.load(pre_fix_root+'fimg.nii.gz')
fix_neuron = nib.load(fix_file_neuron)

# fimg_file to fix_file
output = save_root + 'fimg_fix_affine.mat'
greedy = 'greedy -d 3 -dof 6 -a -m MI -i ' + fix_file + ' ' + fimg_file + ' -o ' + output + ' -ia-image-centers -n 200x100x50'
os.system(greedy)

fimg_fix_affine_align = pre_fix_root+'fimg_fix_affine_aglin.nii.gz'
greedy_reg = 'greedy -d 3 -rf ' + fix_file + ' -rm ' + fimg_file + ' ' + fimg_fix_affine_align + ' -ri LINEAR -r ' + output
os.system(greedy_reg)

fix_data = fix_.get_data()
fix_neuron_data = fix_neuron.get_data()
fix_data_mask = fix_data.mean(2)>0.08
fix_neuron_data_mask = fix_neuron_data.mean(2)>0.1
fix_data_masked = fix_data*fix_data_mask[:, :, None]
fix_neuron_data_mask = fix_neuron_data*fix_neuron_data_mask[:, :, None]

fix_masked = nib.Nifti1Image(fix_data_masked, affine=fix_.affine)
fix_neuron_masked = nib.Nifti1Image(fix_neuron_data_mask, affine=fix_neuron.affine)
nib.save(fix_masked, save_root+'fix_masked.nii.gz')
nib.save(fix_neuron_masked, save_root+'fix_neuron_masked.nii.gz')

fix_file_masked = pre_fix_root+'fix_masked.nii.gz'
fix_file_neuron_masked = pre_fix_root+'fix_neuron_masked.nii.gz'

## from glia to neuron
output = save_root + 'neuron_fix_rigid.mat'
greedy = 'greedy -d 3 -dof 6 -a -m MI -i ' + fix_file_masked + ' ' + fix_file_neuron_masked + ' -o ' + output + ' -ia-image-centers -n 200x100x50'
os.system(greedy)
neuron_fix_rigid_align = pre_fix_root+'atlas_fix_rigid.nii.gz'
greedy_reg = 'greedy -d 3 -rf ' + fix_file + ' -rm ' + fix_file_neuron + ' ' + neuron_fix_rigid_align + ' -ri LINEAR -r ' + output
os.system(greedy_reg)

# the rest of it from neural data registration (already exists)
output_neuron = save_root_neuron + 'atlas_fix_rigid.mat'
shutil.copyfile(output_neuron, save_root + 'neuron_atlas_fix_rigid.mat')

output_neuron = save_root_neuron + 'atlas_fix_affine.mat'
shutil.copyfile(output_neuron, save_root + 'neuron_atlas_fix_affine.mat')

output_neuron = save_root_neuron + 'atlas_fix_wrap_mat.nii.gz'
output_inv_neuron = save_root_neuron + 'atlas_fix_wrap_inv_mat.nii.gz'
shutil.copyfile(output_neuron, save_root + 'neuron_atlas_fix_wrap_mat.nii.gz')
shutil.copyfile(output_inv_neuron, save_root + 'neuron_atlas_fix_wrap_inv_mat.nii.gz')
