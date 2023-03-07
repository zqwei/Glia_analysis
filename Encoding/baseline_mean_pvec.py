from baseline_coding import *


if __name__ == "__main__":
    ind = int(sys.argv[1])
    row = df.iloc[ind]
    save_root = row['save_dir']+'/'
    if not os.path.exists(save_root+'cell_states_mean_baseline_pvec.npy'):
        (CL_CT_swim_mask, OL_CT_swim_mask, 
         CL_CT_dFF, OL_CT_dFF) = process_n_file_data(ind, trial_post = 40, pre_ext = 0)
        CL_CT_ave = CL_CT_dFF.mean(axis=0)
        OL_CT_ave = OL_CT_dFF.mean(axis=0)
        pvec = utest(CL_CT_ave, OL_CT_ave, CL_CT_swim_mask, OL_CT_swim_mask)
        np.save(save_root+'cell_states_mean_baseline_pvec.npy', pvec)
