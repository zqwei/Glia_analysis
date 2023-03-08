from utils import *
from swim_ephys import *

# df = pd.read_csv('../Datalists/data_list_in_analysis_slimmed_v4.csv', index_col=None)
# df = pd.read_csv('../Datalists/data_list_in_analysis_glia_v3.csv', index_col=None)
df = pd.read_csv('../Datalists/data_list_in_analysis_NE_v3.csv', index_col=None)

def kickAssSwimDetect01(ch1,ch2,thre):
    import numpy as np
    # print('processing channel data\n')
    ker = np.exp(-(np.arange(-60,61,1,'f4')**2)/(2*(20**2)));
    ker = ker / ker.sum()
    
    if ch1 is None:
        fltCh1 = np.zeros(ch2.shape)
        peaksIndT1 = np.array([])
    else:
        smch1 = np.convolve(ch1,ker,'same');
        pow1 = (ch1 - smch1)**2;
        fltCh1 = np.convolve(pow1,ker,'same');
        aa1 = np.diff(fltCh1);
        peaksT1 = (aa1[0:-1] > 0) * (aa1[1:] < 0);
        peaksIndT1 = np.argwhere(peaksT1>0).squeeze();
    
    if ch2 is None:
        fltCh2 = np.zeros(ch1.shape)
        peaksIndT2 = np.array([])
    else:    
        smch2 = np.convolve(ch2,ker,'same');
        pow2 = (ch2 - smch2)**2;
        fltCh2 = np.convolve(pow2,ker,'same');
        aa2 = np.diff(fltCh2);
        peaksT2 = (aa2[0:-1] > 0) * (aa2[1:] < 0);
        peaksIndT2 = np.argwhere(peaksT2>0).squeeze();


    x_ = np.arange(0,0.10001,0.00001);
    th1 = np.zeros(fltCh1.size,);
    th2 = np.zeros(fltCh2.size,);
    back1 = np.zeros(fltCh1.size,);
    back2= np.zeros(fltCh2.size,);

    if (len(fltCh1)<360000):
        d_=6000*1
    else:
        d_ = 6000*60;   #% 5 minutes threshold window

    last_i=0

    for i in np.arange(0,fltCh1.size-d_,d_):
        if ch1 is not None:
            peaksIndT1_ = np.argwhere(peaksT1[0:(i+d_)]>0).squeeze();
            a1,_ = np.histogram(fltCh1[peaksIndT1_], x_)
            a1=a1.astype('f4')
            mx1 = (np.argwhere(a1 == a1.max())).min()
            mn1_ind=np.argwhere(a1[0:mx1] < (a1[mx1]/200))
            if (mn1_ind.size>0):
                mn1=mn1_ind.max()
            else:
                mn1=0;
            th1[i:(i+d_+1)] = x_[mx1] + thre*(x_[mx1]-x_[mn1]);
            back1[i:(i+d_+1)] = x_[mx1];
        
        if ch2 is not None:
            peaksIndT2_ = np.argwhere(peaksT2[0:(i+d_)]>0).squeeze();
            a2,_ = np.histogram(fltCh2[peaksIndT2_], x_)
            a2=a2.astype('f4')
            mx2 = (np.argwhere(a2 == a2.max())).min();
            mn2_ind=np.argwhere(a2[0:mx2] < (a2[mx2]/200))
            if (mn2_ind.size>0):
                mn2=mn2_ind.max()
            else:
                mn2=0;
            th2[i:(i+d_+1)] = x_[mx2] + thre*(x_[mx2]-x_[mn2]);
            back2[i:(i+d_+1)] = x_[mx2] ;
        last_i=i

    th1[(last_i+d_+1):] = th1[last_i+d_];
    th2[(last_i+d_+1):] = th2[last_i+d_];
    back1[(last_i+d_+1):] = back1[last_i+d_] ;
    back2[(last_i+d_+1):] = back2[last_i+d_] ;


    # print('\nAssigning bursts and swims\n');
    
    burstBothT = np.zeros(fltCh1.size);
    burstT1=np.zeros(fltCh1.size);
    burstT2=np.zeros(fltCh1.size);
    
    if ch1 is None:
        burstIndT1 = np.array([])
    else:
        burstIndT1 = peaksIndT1[np.argwhere((fltCh1-th1)[peaksIndT1]>0).squeeze()];
        burstT1[burstIndT1]=1;
        burstBothT[burstIndT1] = 1;
    
    if ch2 is None:
        burstIndT2 = np.array([])
    else:    
        burstIndT2 = peaksIndT2[np.argwhere((fltCh2-th2)[peaksIndT2]>0).squeeze()];
        burstT2[burstIndT2]=1;
        burstBothT[burstIndT2] = 2;

    burstBothIndT = np.argwhere(burstBothT>0).squeeze();
    interSwims = np.diff(burstBothIndT);
    swimEndIndB = np.argwhere(interSwims > 600).squeeze();
    swimEndIndB = np.append(swimEndIndB,burstBothIndT.size-1)

    swimStartIndB=0;
    swimStartIndB = np.append(swimStartIndB,swimEndIndB[:-1]+1);
    nonSuperShort = np.argwhere(swimEndIndB != swimStartIndB).squeeze();

    swimEndIndB = swimEndIndB[nonSuperShort];
    swimStartIndB = swimStartIndB[nonSuperShort];

    # swimStartIndB is an index for burstBothIndT
    # burstBothIndT is an idex for time

    swimStartIndT = burstBothIndT[swimStartIndB];
    swimStartT = np.zeros(fltCh1.size);
    swimStartT[swimStartIndT] = 1;

    swimEndIndT = burstBothIndT[swimEndIndB];
    swimEndT = np.zeros(fltCh1.size);
    swimEndT[swimEndIndT] = 1;

    swimdata=dict();
    swimdata['fltCh1']=fltCh1.astype('f4')
    swimdata['fltCh2']=fltCh2.astype('f4')
    swimdata['back1']=back1.astype('f4')
    swimdata['back2']=back2.astype('f4')
    swimdata['th1']=th1.astype('f4')
    swimdata['th2']=th2.astype('f4')
    swimdata['burstBothT']=burstBothT
    swimdata['burstBothIndT']=burstBothIndT
    swimdata['burstIndT1']= burstIndT1
    swimdata['burstIndT2']= burstIndT2
    swimdata['swimStartIndB']= swimStartIndB
    swimdata['swimEndIndB']= swimEndIndB
    swimdata['swimStartIndT']= swimStartIndT
    swimdata['swimEndIndT']= swimEndIndT
    swimdata['swimStartT']= swimStartT
    swimdata['swimEndT']= swimEndT

    return swimdata



for n, row in df.iterrows():
    # if n<60:
    #     continue
    # ephys_root = row['Ephys']
    ephys_root = row['dat_dir']
    # print(ephys_root)
    save_root = row['save_dir']
    ind_ = ephys_root.find('ephys')
    if ind_>0:
        dat_dir = ephys_root[:ind_]
    else:
        dat_dir = ephys_root
    ephys_dir = dat_dir+'/ephys/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if os.path.exists(save_root+'KA_raw_swim.npz'):
        print(n)
        continue
    else:
        print(f'Processing {n}')
    
    ###################################
    ## Downsample sensory and motor input to frames
    ###################################
    if len(glob(ephys_dir+'/*.10chFlt'))==1:
        ephys_dat = glob(ephys_dir+'/*.10chFlt')[0]
        fileContent_ = load(ephys_dat)
    elif len(glob(ephys_dir+'/*.13chFlt'))==1:
        ephys_dat = glob(ephys_dir+'/*.13chFlt')[0]
        fileContent_ = load(ephys_dat, num_channels=13)
        chn = np.ones(13).astype('bool')
        chn[3:6] = False
        fileContent_ = fileContent_[chn]
    else:
        print('check ephys file existence in :'+ephys_dir)
        continue    

    epoch_frame = np.array(fileContent_[5]).astype('int')
    pulse_frame = np.array(fileContent_[8]*1000).astype('int')
    if (pulse_frame>0).sum()==0:
        print('No pulse')
        break
    if (pulse_frame[(epoch_frame%5==3) & (pulse_frame>0)]).sum()==0:
        print('No pulse in pulse periods')
        break
    probe_amp = np.unique(pulse_frame[(epoch_frame%5==3) & (pulse_frame>0)])[0]
    # print(np.unique(pulse_frame[(epoch_frame%5==3) & (pulse_frame>0)]))

    visu_frame = np.array(fileContent_[3])
    visu_frame_ = visu_frame.copy()
    visu_frame_[visu_frame_<0]=0

    ch1 = fileContent_[0]
    ch2 = fileContent_[1]
    
    swim_threshold = 2.0
    is_detected = False
    while not is_detected:
        swimdata = kickAssSwimDetect01(ch1, ch2, swim_threshold)
        fltCh1 = swimdata['fltCh1']
        fltCh2 = swimdata['fltCh2']
        back1 = swimdata['back1']
        back2 = swimdata['back2']
        th1 = swimdata['th1']
        th2 = swimdata['th2']
        burstBothT = swimdata['burstBothT']
        burstBothIndT = swimdata['burstBothIndT']
        burstIndT1 = swimdata['burstIndT1']
        burstIndT2 = swimdata['burstIndT2']
        swimStartIndB = swimdata['swimStartIndB']
        swimEndIndB = swimdata['swimEndIndB']
        swimStartIndT = swimdata['swimStartIndT']
        swimEndIndT = swimdata['swimEndIndT']
        swimStartT = swimdata['swimStartT'] # if swim start at index T
        swimEndT = swimdata['swimEndT']# if swim end at index T
        lswim_frame = np.clip(fltCh1-th1, 0, np.inf)
        rswim_frame = np.clip(fltCh2-th2, 0, np.inf)
        lswim_frame_mean = (lswim_frame>0).mean()
        rswim_frame_mean = (rswim_frame>0).mean()
        if 'simulate-visual' in ephys_root:
            is_detected = True
            break
        if (lswim_frame_mean>0.5) or ((lswim_frame_mean>2*rswim_frame_mean) and (ch2 is not None)):
            print('removing ch1 for swim detection')
            ch1 = None
        elif (rswim_frame_mean>0.5) or ((rswim_frame_mean>2*lswim_frame_mean) and (ch1 is not None)):
            print('removing ch2 for swim detection')
            ch2 = None
        else:
            is_detected = True
    
    swim_t_frame = np.vstack([swimStartIndT,swimEndIndT])
    
    np.savez(save_root+'KA_raw_swim.npz', \
             swimdata=swimdata, \
             probe_amp=probe_amp, \
             swim_t_frame=swim_t_frame, \
             epoch_frame=epoch_frame, \
             lswim_frame=lswim_frame, \
             rswim_frame=rswim_frame, \
             pulse_frame=pulse_frame, \
             visu_frame=visu_frame, \
             visu_frame_=visu_frame_)