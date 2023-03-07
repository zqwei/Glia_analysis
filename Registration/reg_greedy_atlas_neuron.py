from utils import *
import pandas as pd
# df = pd.read_csv('../Datalists/data_list_in_analysis_slimmed_v4.csv')
df = pd.read_csv('../Datalists/data_list_in_analysis_NE_v2.csv')


for ind, row in df.iterrows():
    if ind!=4:
        continue
    if row['high_res']=='None':
        continue
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

    save_root = fimg_dir+'/registration/'
    # if os.path.exists(save_root + 'atlas_fix_wrap_inv_mat.nii.gz'):
    #     print('registration done')
    #     continue

    pre_fix_root = '/nrs/ahrens/Ziqiang/scratch/registration/'+fish_name + '/'
    print(save_root, pre_fix_root)
        
    altas_file = pre_fix_root+'altas.nii.gz'
    fix_file = pre_fix_root+'fix.nii.gz'
    fimg_file = pre_fix_root+'fimg.nii.gz'
    
    # fimg_file to fix_file
    output = save_root + 'fimg_fix_affine.mat'
    if not os.path.exists(output):
        greedy = 'greedy -d 3 -dof 6 -a -m MI -i ' + fix_file + ' ' + fimg_file + ' -o ' + output + ' -ia-image-centers -n 200x100x50'
        os.system(greedy)
    
    fimg_fix_affine_align = pre_fix_root+'fimg_fix_affine_aglin.nii.gz'
    if not os.path.exists(fimg_fix_affine_align):
        greedy_reg = 'greedy -d 3 -rf ' + fix_file + ' -rm ' + fimg_file + ' ' + fimg_fix_affine_align + ' -ri LINEAR -r ' + output
        os.system(greedy_reg)
    
    ## rigid from high-res to atlas
    output = save_root + 'atlas_fix_rigid.mat'
    if not os.path.exists(output):
        greedy = 'greedy -d 3 -dof 6 -a -m MI -i ' + fix_file + ' ' + altas_file + ' -o ' + output + ' -ia-image-centers -n 200x100x50'
        os.system(greedy)
    
    output = save_root + 'atlas_fix_rigid.mat'
    atlas_fix_rigid_align = pre_fix_root+'atlas_fix_rigid.nii.gz'
    if not os.path.exists(atlas_fix_rigid_align):
        greedy_reg = 'greedy -d 3 -rf ' + fix_file + ' -rm ' + altas_file + ' ' + atlas_fix_rigid_align + ' -ri LINEAR -r ' + output
        os.system(greedy_reg)
    
    ## affine
    rigid_output = save_root + 'atlas_fix_rigid.mat'
    output = save_root + 'atlas_fix_affine.mat'
    if not os.path.exists(output):
        greedy = 'greedy -d 3 -dof 12 -a -m NMI -i ' + fix_file + ' ' + altas_file + ' -o ' + output + ' -ia ' + rigid_output + ' -n 50x20x10'
        os.system(greedy)
    
    atlas_fix_affine_align = pre_fix_root+'atlas_fix_affine.nii.gz'
    output = save_root + 'atlas_fix_affine.mat'
    if not os.path.exists(atlas_fix_affine_align):
        greedy_reg = 'greedy -d 3 -rf ' + fix_file + ' -rm ' + altas_file + ' ' + atlas_fix_affine_align + ' -ri LINEAR -r ' + output
        os.system(greedy_reg)
    
    ## deformable
    affine_output = save_root + 'atlas_fix_affine.mat'
    output = save_root + 'atlas_fix_wrap_mat.nii.gz'
    output_inv = save_root + 'atlas_fix_wrap_inv_mat.nii.gz'
    smoothing = '1.7vox 0.7vox' # Deformable registration smoothing parameters
    if not os.path.exists(output):
        greedy = 'greedy -d 3 -m NCC 12x12x6 -i ' + fix_file + ' ' + altas_file + ' -it ' + affine_output + ' -o ' + output + ' -n 100x40x20 -s ' + smoothing + ' -oinv ' + output_inv
        os.system(greedy)
    
    atlas_fix_wrap_align = pre_fix_root+'atlas_fix_wrap.nii.gz'
    if not os.path.exists(atlas_fix_wrap_align):
        greedy = 'greedy -d 3 -rf ' + fix_file + ' -rm ' + altas_file + ' ' + atlas_fix_wrap_align + ' -ri LINEAR -r ' + output + ' ' + affine_output
        os.system(greedy)
    # invert
    atlas_fix_affine_wrap_inv = pre_fix_root+'atlas_fix_wrap_inv.nii.gz'
    if not os.path.exists(atlas_fix_affine_wrap_inv):
        greedy = 'greedy -d 3 -rf ' + altas_file + ' -rm ' + fix_file + ' ' + atlas_fix_affine_wrap_inv + ' -r ' + affine_output + ',-1 ' + output_inv
        os.system(greedy)
    
    # clean up folder
    # clean_folder = f'rm -rf {pre_fix_root}*.*'
    # os.system(clean_folder)