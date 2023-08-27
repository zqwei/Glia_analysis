from utils import *
df = pd.read_csv('../Processing/data_list_in_analysis_osc_curated.csv')
# df = pd.read_csv('../Processing/data_list_in_analysis_downsample.csv')

for ind, row in df.iterrows():
    if ind==0:
        continue
    save_root = row['save_dir']+'/'
    print(row['taskType'])
    print(save_root)

    dFF_ = np.load(save_root+'cell_dff.npz',allow_pickle=True)['dFF']
    num_dff = dFF_.shape[-1]
    num_cells = dFF_.shape[0]

    _ = np.load(save_root+'ephys.npz', allow_pickle=True)
    probe_amp   = _['probe_amp']
    probe_gain  = _['probe_gain']
    swim_t_frame= _['swim_t_frame']
    epoch_frame = _['epoch_frame']
    lswim_frame = _['lswim_frame']
    rswim_frame = _['rswim_frame']
    pulse_frame = _['pulse_frame']
    visu_frame  = _['visu_frame']
    visu_frame_ = _['visu_frame_']
    swim_channel = np.array2string(_['swim_channel']).replace("'", "")
    swim_threshold = _['swim_threshold']
    pulse_amp = probe_amp
    
    swim_frame_power  = lswim_frame + rswim_frame
    if 'left' in swim_channel:
        swim_frame_power  = lswim_frame 
    if 'right' in swim_channel:
        swim_frame_power  = rswim_frame 
    swim_frame_med = medfilt(swim_frame_power, 201)
    swim_frame_power = swim_frame_power - swim_frame_med

    swim_ons, swim_offs = swim_t_frame
    swim_frame_ = np.zeros(len(swim_frame_power)).astype('bool')
    for swim_on, swim_off in zip(swim_ons, swim_offs):
        swim_frame_[swim_on:swim_off]=1
    swim_frame = swim_frame_.copy()

    pulse_epoch_on = np.where((epoch_frame[:-1]%5==2) & (epoch_frame[1:]%5==3))[0]
    pulse_epoch_off = np.where((epoch_frame[:-1]%5==3) & (epoch_frame[1:]%5!=3))[0]
    if len(pulse_epoch_off)<len(pulse_epoch_on):
        pulse_epoch_on = pulse_epoch_on[:len(pulse_epoch_off)]
    elif len(pulse_epoch_off)>len(pulse_epoch_on):
        print('pulse length error!')
    
    mlt_pulse_on = []
    mlt_nopulse_on = []
    mlt_pulse_trial = []
    mlt_pulse_type = []
    mlt_pulse_swim = []
    mlt_pulse_swim_on = []

    for n_pen, n_pef in zip(pulse_epoch_on, pulse_epoch_off):
        _ = np.where(pulse_frame[n_pen:n_pef]==pulse_amp)[0]
        if len(_)>0:
            mlt_pulse_on.append(n_pen+_[0])
        else:
            mlt_nopulse_on.append(n_pen+1)

    t_post = (pulse_epoch_off - pulse_epoch_on).min()+10
    t_pre = 5
    for n, trial in enumerate(mlt_pulse_on):
        swim_ = swim_frame[trial-t_pre:trial+t_post]
        mlt_pulse_swim.append(swim_)
        if swim_.sum()==0:
            mlt_pulse_swim_on.append(t_post+t_pre+10)
        else:
            mlt_pulse_swim_on.append(np.where(swim_)[0][0])
        mlt_pulse_trial.append(True)
        mlt_pulse_type.append(epoch_frame[trial]//5)

    for n, trial in enumerate(mlt_nopulse_on):
        swim_ = swim_frame[trial-t_pre:trial+t_post]
        mlt_pulse_swim.append(swim_)
        if swim_.sum()==0:
            mlt_pulse_swim_on.append(t_post+t_pre+10)
        else:
            mlt_pulse_swim_on.append(np.where(swim_)[0][0])
        mlt_pulse_trial.append(False)
        mlt_pulse_type.append(epoch_frame[trial]//5)

    mlt_pulse_swim = np.array(mlt_pulse_swim)
    mlt_pulse_swim_on = np.array(mlt_pulse_swim_on)
    mlt_pulse_trial = np.array(mlt_pulse_trial)
    mlt_pulse_type = np.array(mlt_pulse_type)
    mlt_nopulse_on = np.array(mlt_nopulse_on)
    mlt_pulse_on = np.array(mlt_pulse_on)

    swim_nopulse_on = mlt_pulse_swim_on[mlt_pulse_trial==0]-5
    swim_pulse_on = mlt_pulse_swim_on[mlt_pulse_trial==1]-5
    mlt_pulse_on_dff = np.zeros((num_cells, t_pre+t_post))
    mlt_nopulse_on_dff = np.zeros((num_cells, t_pre+t_post))
    type_nopulse_on = mlt_pulse_type[mlt_pulse_trial==0]==0
    type_pulse_on = mlt_pulse_type[mlt_pulse_trial==1]==0
    mlt_pulse_on_act = np.zeros((num_cells, t_pre+t_post))
    mlt_nopulse_on_act = np.zeros((num_cells, t_pre+t_post))
    mlt_pulse_on_pss = np.zeros((num_cells, t_pre+t_post))
    mlt_nopulse_on_pss = np.zeros((num_cells, t_pre+t_post))

    print(-t_pre, t_post)
    for n, t in enumerate(range(-t_pre, t_post)):
        print(n)
        mlt_pulse_on_dff[:, n] = dFF_[:, mlt_pulse_on[swim_pulse_on>=t]+t].mean(axis=1)
        mlt_nopulse_on_dff[:, n] = dFF_[:, mlt_nopulse_on[swim_nopulse_on>=t]+t].mean(axis=1)
        mlt_pulse_on_act[:, n] = dFF_[:, mlt_pulse_on[(swim_pulse_on>=t) & type_pulse_on]+t].mean(axis=1)
        mlt_nopulse_on_act[:, n] = dFF_[:, mlt_nopulse_on[(swim_nopulse_on>=t) & type_nopulse_on]+t].mean(axis=1)
        mlt_pulse_on_pss[:, n] = dFF_[:, mlt_pulse_on[(swim_pulse_on>=t) & ~type_pulse_on]+t].mean(axis=1)
        mlt_nopulse_on_pss[:, n] = dFF_[:, mlt_nopulse_on[(swim_nopulse_on>=t) & ~type_nopulse_on]+t].mean(axis=1)

    np.savez(save_root+'pulse_dff', \
             mlt_pulse_on_dff=mlt_pulse_on_dff, \
             mlt_nopulse_on_dff=mlt_nopulse_on_dff, \
             mlt_pulse_on_act=mlt_pulse_on_act, \
             mlt_nopulse_on_act=mlt_nopulse_on_act, \
             mlt_pulse_on_pss=mlt_pulse_on_pss, \
             mlt_nopulse_on_pss=mlt_nopulse_on_pss, \
             mlt_pulse_swim=mlt_pulse_swim, \
             mlt_pulse_swim_on=mlt_pulse_swim_on, \
             mlt_pulse_trial=mlt_pulse_trial, \
             mlt_pulse_type=mlt_pulse_type, \
             mlt_nopulse_on=mlt_nopulse_on, \
             mlt_pulse_on=mlt_pulse_on)