#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

import sys
from sensory_motor_bc import *
from brain_state_bcr import *
from brain_state_rl_bcr import *
from multi_vel_bc import *
from pathlib import Path
from sensory_motor_pvi_task import *
from brain_state_pvi_bcr import *
df = pd.read_csv('../Processing/data_list_in_analysis_downsample.csv')
# df = pd.read_csv('../Processing/data_list_in_analysis_osc_curated.csv')

    
if __name__ == "__main__":
    row = df.iloc[int(sys.argv[1])]
    print(row)
    sensory_motor_bar_code(row)
    brain_state_bar_code_raw(row)
    
'''
for (( n=0; n<=10; n++ ))
do
bsub -n 4 -J DS$n -o /dev/null "/groups/ahrens/home/weiz/Projects/Glia_analysis/Notebooks/run_bar_code_cluster.py $n > output_ds$n" 
done
'''
