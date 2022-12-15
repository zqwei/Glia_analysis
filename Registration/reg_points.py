from utils import *
import pandas as pd
def read_reg_mat(file):
    with open(file, 'r') as f:
        l = [[float(num) for num in line.replace(' \n', '').split(' ')] for line in f]
    return np.array(l)

df = pd.read_csv('../Processing/data_list_in_analysis_osc_curated.csv')

for ind, row in df.iterrows():
    moving_root = row['dat_dir']
    fimg_dir = row['save_dir']
    save_root = fimg_dir+'/registration/'
    write_path =  fimg_dir+'/registration/fixed_reference.h5'
    if not os.path.exists(write_path):
        print('No high res file')
    else:
        fimg = np.load(fimg_dir + 'Y_ave.npy').squeeze().astype('float')
        fix = read_h5(write_path).astype('float')

    
    fimg_smooth = fimg.copy()
    thresh_u = np.percentile(fimg, 99)
    thresh_l = np.percentile(fimg, 10)
    fimg_smooth[fimg_smooth>thresh_u] = thresh_u
    fimg_smooth = fimg_smooth - thresh_l
    fimg_smooth[fimg_smooth<0] = 0
    fimg_smooth = fimg_smooth/fimg_smooth.max()
    fimg = None

    fatlas_smooth = atlas.copy()
    thresh_u = np.percentile(fatlas_smooth, 99)
    thresh_l = np.percentile(fatlas_smooth, 10)
    fatlas_smooth[fatlas_smooth>thresh_u] = thresh_u
    fatlas_smooth = fatlas_smooth - thresh_l
    fatlas_smooth[fatlas_smooth<0] = 0
    fatlas_smooth = fatlas_smooth/fatlas_smooth.max()
    atlas = None

    ffix_smooth = fix.copy()
    thresh_u = np.percentile(ffix_smooth, 99)
    thresh_l = np.percentile(ffix_smooth, 10)
    ffix_smooth[ffix_smooth>thresh_u] = thresh_u
    ffix_smooth = ffix_smooth - thresh_l
    ffix_smooth[ffix_smooth<0] = 0
    ffix_smooth = ffix_smooth/ffix_smooth.max()
    fix = None
    
    