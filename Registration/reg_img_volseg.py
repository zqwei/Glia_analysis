from utils import *
import pandas as pd
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


for ind, row in df.iterrows():
    if ind<79:
        continue
    moving_root = row['im_volseg']
    fimg_dir = row['save_dir']
    fish_name = fimg_dir.split('/')[-2]
    if 'im_CM0_voluseg' in fimg_dir:
        fish_name = fimg_dir.split('/')[-3]+'_im_CM0_voluseg'
    if 'im_CM1_voluseg' in fimg_dir:
        fish_name = fimg_dir.split('/')[-3]+'_im_CM1_voluseg'
    if 'im_CM0_dff' in fimg_dir:
        fish_name = fimg_dir.split('/')[-3]+'_im_CM0_dff'
    if 'im_CM1_dff' in fimg_dir:
        fish_name = fimg_dir.split('/')[-3]+'_im_CM1_dff'
    save_root = fimg_dir+'/registration/'
    write_path = row['high_res']
    ind_ = write_path.find('/processed/')
    fixed_root = write_path[:ind_+1]
    if not os.path.exists(write_path):
        print(write_path)
        print('No high res file')
    if os.path.exists(save_root + 'atlas_fix_wrap_inv_mat.nii.gz'):
        print('registration done')
    
    print(fish_name)
    
    fimg = np.load(fimg_dir + 'Y_ave.npy').squeeze().astype('float')
    fix = read_h5(write_path).astype('float')

    fimg_smooth = fimg.copy()
    thresh_u = np.percentile(fimg, 99)
    thresh_l = np.percentile(fimg, 75)
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

    if 'voluseg' in moving_root:
        moving_root = moving_root[:-8]
    print(moving_root)
    if os.path.exists(moving_root + '/im_CM0/ch0.xml'):
        cinfo = getCameraInfo(moving_root + '/im_CM0/ch0.xml')
        fimg_zvox = float(cinfo['z_step'])
    if os.path.exists(moving_root + '/im_CM1/ch0.xml'):
        cinfo = getCameraInfo(moving_root + '/im_CM1/ch0.xml')
        fimg_zvox = float(cinfo['z_step'])
    if os.path.exists(moving_root + '/ch0.xml'):
        cinfo = getCameraInfo(moving_root + '/ch0.xml')
        fimg_zvox = float(cinfo['z_step'])

    fix_zvox = 1.
    if os.path.exists(fixed_root + '/im_CM0/ch0.xml'):
        cinfo = getCameraInfo(fixed_root + '/im_CM0/ch0.xml')
        fix_zvox = float(cinfo['z_step'])
    if os.path.exists(fixed_root + '/im_CM1/ch0.xml'):
        cinfo = getCameraInfo(fixed_root + '/im_CM1/ch0.xml')
        fix_zvox = float(cinfo['z_step'])
    if os.path.exists(fixed_root + '/ch0.xml'):
        cinfo = getCameraInfo(fixed_root + '/ch0.xml')
        fix_zvox = float(cinfo['z_step'])

    atlas_zvox = 2.
    spim_res = 0.406
    spim_res_ = 0.406
    if 'voluseg' in fish_name:
        spim_res_ = 0.406*2
    if 'dff' in fish_name:
        spim_res_ = 0.406*10
    fimg_vox = np.array([spim_res_, spim_res_, fimg_zvox, 1])
    fix_vox = np.array([spim_res, spim_res, fix_zvox, 1])
    atlas_vox = np.array([spim_res, spim_res, atlas_zvox, 1])

    print(fimg_vox, fix_vox, atlas_vox)
    print(ffix_smooth.shape, fimg_smooth.shape)
    
    fimg_smooth_ = fimg_smooth[:,:,::-1] # flip
    ffix_smooth_ = ffix_smooth[:,:,::-1] # flip
    fatlas_smooth_ = fatlas_smooth[:170, :, :]
    altas_ = nib.Nifti1Image(fatlas_smooth_.swapaxes(0, 2), affine=np.diag(atlas_vox))
    fix_ = nib.Nifti1Image(ffix_smooth_.swapaxes(0, 2), affine=np.diag(fix_vox))
    fimg_ = nib.Nifti1Image(fimg_smooth_.swapaxes(0, 2), affine=np.diag(fimg_vox))
    pre_fix_root = '/nrs/ahrens/Ziqiang/scratch/registration/'+fish_name + '/'
    if not os.path.exists(pre_fix_root):
        os.makedirs(pre_fix_root)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    nib.save(altas_, pre_fix_root+'altas.nii.gz')
    nib.save(fix_, pre_fix_root+'fix.nii.gz')
    nib.save(fimg_, pre_fix_root+'fimg.nii.gz')

    fz, fy, fx = ffix_smooth_.shape
    az, ay, ax = fatlas_smooth_.shape
    print((fz, fy, fx), (az, ay, ax))
    np.savez(save_root+'sample_parameters', fix_range=np.array([[ 0, fx],[0, fy], [0, fz]]), \
                                            atlas_range=np.array([[0, ax], [ 0, ay], [0, 170]]), \
                                            fimg_vox = fimg_vox, fix_vox = fix_vox, atlas_vox = atlas_vox, \
                                            flip_xyz = np.array([1, 0, 0]).astype('bool'))  # flip
    print(pre_fix_root)