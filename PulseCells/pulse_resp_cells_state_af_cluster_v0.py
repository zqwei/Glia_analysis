import numpy as np
import pandas as pd
import numpy.ma as ma
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
def get_cell_in_map(cells_center, result_tmp=result_tmp):
    num_cells = cells_center.shape[0]
    cell_in_map = np.zeros(num_cells).astype('bool')
    for n in range(num_cells):
        z, y, x = np.around(cells_center[n]).astype('int')
        cell_in_map[n] = result_tmp[z-3:z+3, y-3:y+3, x-3:x+3].sum()>0
    return cell_in_map


df = pd.read_csv('../Datalists/data_list_in_analysis_neuron_v0.csv')
mask_folder = '/nrs/ahrens/Ziqiang/Jing_Glia_project/cell_masks/'
_ = np.load(mask_folder + 'pulse_state_cell_mask.npz', allow_pickle=True)
result_tmp = _['result_tmp']
save_cluster_root = '/nrs/ahrens/Ziqiang/Jing_Glia_project/cell_clusters_af/'

## Cell location & dynamics collection
cell_locs = []
CL_dFF_list = []
OL_dFF_list = []
CL_dFF_list_ = []
OL_dFF_list_ = []
animal_indx = []

for ind, row in df.iterrows():
    if ind>5:
        continue
    save_root = row['save_dir']+'/'
    cells_center = np.load(save_root+'cell_center_registered.npy')
    cell_in_brain = np.load(save_root+'cell_in_brain.npy')
    cell_idx = np.load(save_root + 'cell_state_pulse_filtered.npy')
    cells_center = cells_center[cell_in_brain][cell_idx]
    cell_in_map = get_cell_in_map(cells_center, result_tmp=result_tmp)
    if cell_in_map.sum()>0:
        cell_locs.append(cells_center[cell_in_map])
    
    trial_post = 35
    trial_pre = 5
    pre_ext = 0
    swim_thres = 30
    _ = np.load(save_root + 'KA_ephys.npz', allow_pickle=True)
    probe_amp=_['probe_amp']
    swim_t_frame=_['swim_t_frame']
    dFF_ = np.load(save_root+'cell_dff.npz', allow_pickle=True)['dFF'][cell_in_brain][cell_idx][cell_in_map]
    len_dff = dFF_.shape[1]
    epoch_frame=_['epoch_frame'][:len_dff]
    pulse_frame=_['pulse_frame'][:len_dff]
    visu_frame=_['visu_frame'][:len_dff]
    lswim_frame=_['lswim_frame'][:len_dff]
    rswim_frame=_['rswim_frame'][:len_dff]
    visu_frame_=_['visu_frame_'][:len_dff]
    
    CL_idx = epoch_frame<=1
    rl = row['rl']
    rr = row['rr']
    if rl >= rr:
        swim_frame_ = lswim_frame
    else:
        swim_frame_ = rswim_frame
    pulse_frame_=pulse_frame.copy()
    pulse_frame_[epoch_frame%5<3]=0
    
    # 0 reset, 1 evoke, 2 pause, 3 probe, 4 reset
    epoch_on = np.where((epoch_frame[1:]%5==0) & (epoch_frame[:-1]%5>0))[0]+1
    len_ = len(epoch_on)-1
    dFF_trial = []
    swim_mask_trial = []
    CL_trial = []

    for n_ in range(len_):
        on_ = epoch_on[n_]
        off_ = epoch_on[n_+1]-1
        # check if this is a complete trial
        # skip incomplete trials
        if len(np.unique(epoch_frame[on_:off_]))<5:
            continue
        epoch_ = epoch_frame[on_:off_]
        # remove the trials without swim during evoke
        if swim_frame_[on_:off_][epoch_%5==1].sum()==0:
            continue
        swm_ = swim_frame_[on_:off_]
        pulse_ = pulse_frame_[on_:off_]
        CL_trial_ = epoch_[10]//5==0

        # remove OL active trials -- fish swim during pause
        if (not CL_trial_) & (swim_frame_[on_:off_][epoch_%5==2].sum()>0):
            continue

        # remove CL passive trials
        if swim_frame_[on_:off_][epoch_%5==2].sum()==0:
            last_swm = np.where(swim_frame_[on_:off_][epoch_%5<2])[0][-1]
            last_swm = last_swm - (epoch_%5<2).sum()
        else:
            last_swm = 0
        if CL_trial_ & (last_swm<-10):
            continue

        # length of pause -- it seems like some of them are extremely long
        # remove the long pause trial
        if (not CL_trial_) & ((epoch_%5==2).sum()>10):
            continue
        if (CL_trial_) & ((epoch_%5==2).sum()<20):
            continue
        # print((epoch_%5==2).sum())
        if ((epoch_%5==2).sum()>100):
            continue

        # set probe on time
        catch_trial_ = pulse_.sum()==0
        # pause trial
        if catch_trial_:
            # probe_on_ = (epoch_<3).sum()
            continue
        else:
            probe_on_ = np.where(pulse_>0)[0][0]

        probe_on_ = probe_on_+on_

        swm_ = np.cumsum(swim_frame_[probe_on_-trial_pre:probe_on_+trial_post])
        if (swm_>0).sum()>swim_thres:
            continue

        # catch_trial.append(catch_trial_)
        CL_trial.append(CL_trial_)
        dFF_trial.append(dFF_[:, probe_on_-trial_pre:probe_on_+trial_post])
        swim_mask_trial.append(swm_<=0)
        
    dFF_trial = np.array(dFF_trial)
    dFF_trial = dFF_trial.transpose([1, 2, 0])
    dFF_trial_ = dFF_trial - dFF_trial[:, :trial_pre, :].mean(axis=1, keepdims=True)
    swim_mask_trial = np.array(swim_mask_trial).T
    # catch_trial = np.array(catch_trial)
    CL_trial = np.array(CL_trial)
    
    dat_ = dFF_trial[:, :, CL_trial]
    swim_mask_ = swim_mask_trial[:, CL_trial]
    swim_mask_ = np.broadcast_to(swim_mask_, dat_.shape)
    mask_arr = ma.masked_array(dat_, mask =~swim_mask_)
    CL_dFF_list.append(mask_arr.mean(axis=-1))

    dat_ = dFF_trial[:, :, ~CL_trial]
    swim_mask_ = swim_mask_trial[:, ~CL_trial]
    swim_mask_ = np.broadcast_to(swim_mask_, dat_.shape)
    mask_arr = ma.masked_array(dat_, mask =~swim_mask_)
    OL_dFF_list.append(mask_arr.mean(axis=-1))
    
    
    dat_ = dFF_trial_[:, :, CL_trial]
    swim_mask_ = swim_mask_trial[:, CL_trial]
    swim_mask_ = np.broadcast_to(swim_mask_, dat_.shape)
    mask_arr = ma.masked_array(dat_, mask =~swim_mask_)
    CL_dFF_list_.append(mask_arr.mean(axis=-1))

    dat_ = dFF_trial_[:, :, ~CL_trial]
    swim_mask_ = swim_mask_trial[:, ~CL_trial]
    swim_mask_ = np.broadcast_to(swim_mask_, dat_.shape)
    mask_arr = ma.masked_array(dat_, mask =~swim_mask_)
    OL_dFF_list_.append(mask_arr.mean(axis=-1))
    
    animal_indx.append([ind]*dFF_trial.shape[0])

CL_dFF_list = np.concatenate(CL_dFF_list, axis=0)
OL_dFF_list = np.concatenate(OL_dFF_list, axis=0)
CL_dFF_list_ = np.concatenate(CL_dFF_list_, axis=0)
OL_dFF_list_ = np.concatenate(OL_dFF_list_, axis=0)
cell_locs = np.concatenate(cell_locs)
animal_indx_ = np.concatenate(animal_indx)

## saving collected data
np.savez(save_cluster_root+'cell_list.npz', \
         CL_dFF_list=CL_dFF_list, \
         OL_dFF_list=OL_dFF_list, \
         CL_dFF_list_=CL_dFF_list_, \
         OL_dFF_list_=OL_dFF_list_, \
         cell_locs=cell_locs, \
         animal_indx_=animal_indx_)


## Cell location based clustering
# rescale z
cell_locs_ = cell_locs.copy()
cell_locs_[:, 0] = cell_locs_[:, 0]
clustering = DBSCAN(eps=20, n_jobs=-1).fit(cell_locs_)

labels = clustering.labels_.copy()
idx_, cnt_ = np.unique(clustering.labels_, return_counts=True)
for idx in idx_:
    if idx<0:
        continue
    if (labels==idx).sum()<20:
        labels[labels==idx] = -1
ulabels, cnt_ = np.unique(labels, return_counts=True)
cnt_ = cnt_[ulabels>=0]
ulabels = ulabels[ulabels>=0]
ulabels = ulabels[np.argsort(-cnt_)]
labels_ = labels.copy()

for n, idx in enumerate(ulabels):
    labels_[labels==idx] = n
spatial_labels = labels_

c_thres = 17
slabels_cleaned = spatial_labels.copy()
slabels_cleaned[slabels_cleaned>c_thres] = -1

slabels_cleaned_redefined = slabels_cleaned.copy()
slabels_cleaned_redefined[slabels_cleaned==2] = 1 # tectum
slabels_cleaned_redefined[slabels_cleaned==4] = 3 # forebrain
slabels_cleaned_redefined[slabels_cleaned==15] = 3
slabels_cleaned_redefined[slabels_cleaned==8] = 5 # IO
slabels_cleaned_redefined[slabels_cleaned==7] = 6 # PT
slabels_cleaned_redefined[slabels_cleaned==13] = 9 # SloMO
slabels_cleaned_redefined[slabels_cleaned==12] = 11 # TL
slabels_cleaned_redefined[slabels_cleaned==16] = 14 # tectum cell bodies

np.savez(save_cluster_root+'cell_spatial_labels.npz', \
         spatial_labels = spatial_labels, \
         slabels_cleaned = slabels_cleaned, \
         slabels_cleaned_redefined = slabels_cleaned_redefined)


## Cell dynamics based clustering
trial_all = np.concatenate([CL_dFF_list_, OL_dFF_list_], axis=1)
idxx = (animal_indx_<3) & (slabels_cleaned>=0)
dat_ =trial_all[idxx].data
zdat_ =  zscore(dat_, axis=-1)
clustering = AgglomerativeClustering(n_clusters=40).fit(zdat_)
func_labels = clustering.labels_
cluster_act_mat = np.array([dat_[func_labels==n].mean(axis=0) for n in range(40)])
corr_, p_ = spearmanr(cluster_act_mat, axis=1)
corr_[np.eye(40)==1]=-1
labels = func_labels.copy()
for nlabel in range(40):
    similar_labels = np.where(corr_[nlabel]>0.7)[0]
    if len(similar_labels)>1:
        for mlabel in similar_labels:
            labels[labels==mlabel] = nlabel
        corr_[similar_labels] = np.nan
    if (labels==nlabel).sum()>0:
        if len(np.unique(animal_indx_[idxx][labels==nlabel]))<3:
               labels[labels==nlabel] = np.argmax(corr_[nlabel])
ulabel, cnt = np.unique(labels, return_counts=True)
ulabel_sort = ulabel[np.argsort(-cnt)]
num_labels = len(ulabel_sort)
labels_resort = labels.copy()
for n in range(num_labels):
    labels_resort[labels == ulabel_sort[n]] = n
np.savez(save_cluster_root+'cell_func_labels.npz', \
         idxx = idxx,\
         labels_resort = labels_resort)

## Combining two clusters
idxx = (animal_indx_<3) & (slabels_cleaned>=0)
func_labels = labels_resort.copy()
sp_labels = slabels_cleaned_redefined[idxx]
func_sp_labels = func_labels*100 + sp_labels
thres_ = 20
ulabel, cnt = np.unique(func_sp_labels, return_counts=True)
func_sp_labels_sorted = -np.ones(func_sp_labels.shape)
sindx = np.argsort(-cnt)
ulabel = ulabel[sindx]
cnt = cnt[sindx]
n_labels = len(ulabel)
m = 0
for n in range(n_labels):
    if cnt[n]<thres_:
        continue
    func_sp_labels_sorted[func_sp_labels== ulabel[n]] = m
    m = m+1
    
np.save(save_cluster_root+'func_sp_labels_sorted.npy',func_sp_labels_sorted)
